"""MPE-style fused Metal particle environments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .device import resolve_device, seed_device
from .native import shader_source
from .tensor_env import TensorEnv, TensorSpec
from .types import StepResult


@dataclass(frozen=True)
class MPESimpleConfig:
    """Static config for an MPE Simple-style particle task.

    This is intentionally the first small public benchmark target: one agent,
    one landmark, batched slots, no Python objects in the hot path.
    """

    num_envs: int = 65536
    world_size: float = 1.0
    dt: float = 0.1
    max_cycles: int = 25
    accel: float = 5.0
    damping: float = 0.25
    max_speed: float = 1.0


class MetalMPESimplePool(TensorEnv):
    """Batched MPE Simple-style env with a fused Metal step on MPS.

    Each slot has one movable agent and one static landmark. The observation is
    ``[vx, vy, landmark_rel_x, landmark_rel_y]``. Continuous actions use the
    familiar MPE 5-channel layout ``[noop, left, right, down, up]`` and are
    clamped to ``[0, 1]`` before integration.
    """

    def __init__(
        self,
        cfg: MPESimpleConfig | None = None,
        *,
        device: str | torch.device = "auto",
        use_shader: bool = True,
    ) -> None:
        self.cfg = cfg or MPESimpleConfig()
        if self.cfg.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.cfg.max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        self.num_envs = self.cfg.num_envs
        self.device = resolve_device(device)
        self.dtype = torch.float32
        self.single_observation_spec = TensorSpec(
            shape=(4,),
            dtype=self.dtype,
            low=-float("inf"),
            high=float("inf"),
        )
        self.single_action_spec = TensorSpec(
            shape=(5,),
            dtype=self.dtype,
            low=0.0,
            high=1.0,
        )
        self.agent = torch.empty((self.cfg.num_envs, 4), device=self.device, dtype=self.dtype)
        self.landmark = torch.empty((self.cfg.num_envs, 2), device=self.device, dtype=self.dtype)
        self.reward = torch.empty((self.cfg.num_envs,), device=self.device, dtype=self.dtype)
        self.terminated = torch.empty((self.cfg.num_envs,), device=self.device, dtype=torch.bool)
        self.truncated = torch.empty((self.cfg.num_envs,), device=self.device, dtype=torch.bool)
        self.steps = torch.empty((self.cfg.num_envs,), device=self.device, dtype=torch.int32)
        self._pending_actions: torch.Tensor | None = None
        self._shader = _mpe_simple_shader() if use_shader and self.device.type == "mps" else None
        self.reset()

    @property
    def using_shader(self) -> bool:
        return self._shader is not None

    def reset(self, *, seed: int | None = None, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if seed is not None:
            seed_device(seed, self.device)
        ids = self._env_ids(env_ids)
        n = int(ids.numel())
        self.agent[ids, :2] = self._rand_pos((n, 2))
        self.agent[ids, 2:] = 0
        self.landmark[ids] = self._rand_pos((n, 2))
        self.reward[ids] = 0
        self.terminated[ids] = False
        self.truncated[ids] = False
        self.steps[ids] = 0
        return self.observe()

    def reset_done(self, done: torch.Tensor) -> torch.Tensor:
        mask = done.to(device=self.device, dtype=torch.bool)
        if tuple(mask.shape) != (self.cfg.num_envs,):
            raise ValueError(f"expected done shape {(self.cfg.num_envs,)}, got {tuple(mask.shape)}")
        mask2 = mask.unsqueeze(1)
        new_pos = self._rand_pos((self.cfg.num_envs, 2))
        new_landmark = self._rand_pos((self.cfg.num_envs, 2))
        zeros2 = torch.zeros((self.cfg.num_envs, 2), device=self.device, dtype=self.dtype)
        self.agent[:, :2] = torch.where(mask2, new_pos, self.agent[:, :2])
        self.agent[:, 2:] = torch.where(mask2, zeros2, self.agent[:, 2:])
        self.landmark[:] = torch.where(mask2, new_landmark, self.landmark)
        self.reward[:] = torch.where(mask, torch.zeros_like(self.reward), self.reward)
        self.terminated[:] = torch.where(mask, torch.zeros_like(self.terminated), self.terminated)
        self.truncated[:] = torch.where(mask, torch.zeros_like(self.truncated), self.truncated)
        self.steps[:] = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        return self.observe()

    def observe(self) -> torch.Tensor:
        rel_landmark = self.landmark - self.agent[:, :2]
        return torch.cat([self.agent[:, 2:], rel_landmark], dim=1)

    def send(self, actions: torch.Tensor) -> None:
        self._pending_actions = self._validate_actions(actions)

    def recv(self) -> StepResult:
        if self._pending_actions is None:
            raise RuntimeError("recv called before send")
        actions = self._pending_actions
        self._pending_actions = None
        return self.step(actions)

    def step(self, actions: torch.Tensor) -> StepResult:
        actions = self._validate_actions(actions)
        if self._shader is None:
            self._torch_step(actions)
        else:
            cfg = self.cfg
            self._shader.mpe_simple_step(
                self.agent,
                actions,
                self.landmark,
                self.reward,
                self.terminated,
                self.truncated,
                self.steps,
                cfg.num_envs,
                cfg.dt,
                cfg.accel,
                cfg.damping,
                cfg.max_speed,
                cfg.max_cycles,
            )
        info = {"steps": self.steps, "using_shader": torch.tensor(int(self.using_shader), device=self.device)}
        return StepResult(
            obs=self.observe(),
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info=info,
        )

    def sample_random_actions(self) -> torch.Tensor:
        return self._rand(self.action_shape)

    def zero_actions(self) -> torch.Tensor:
        return torch.zeros(self.action_shape, device=self.device, dtype=self.dtype)

    def _torch_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        active = ~(self.terminated | self.truncated)
        action = actions.clamp(0.0, 1.0)
        force = torch.stack((action[:, 2] - action[:, 1], action[:, 4] - action[:, 3]), dim=1)
        vel = (self.agent[:, 2:] * (1.0 - cfg.damping)) + (force * cfg.accel * cfg.dt)
        speed = torch.linalg.vector_norm(vel, dim=1, keepdim=True).clamp_min(1e-8)
        vel = vel * torch.clamp(cfg.max_speed / speed, max=1.0)
        pos = self.agent[:, :2] + (vel * cfg.dt)
        active2 = active.unsqueeze(1)
        self.agent[:, :2] = torch.where(active2, pos, self.agent[:, :2])
        self.agent[:, 2:] = torch.where(active2, vel, self.agent[:, 2:])
        self.steps += active.to(torch.int32)

        delta = self.landmark - self.agent[:, :2]
        dist = torch.linalg.vector_norm(delta, dim=1)
        self.terminated[:] = False
        self.truncated[:] = self.truncated | (self.steps >= cfg.max_cycles)
        self.reward[:] = torch.where(active, -dist, torch.zeros_like(dist))

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, device=self.device, dtype=self.dtype)

    def _rand_pos(self, shape: tuple[int, ...]) -> torch.Tensor:
        return ((self._rand(shape) * 2.0) - 1.0) * self.cfg.world_size

    def _env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.cfg.num_envs, device=self.device, dtype=torch.long)
        out = env_ids.to(device=self.device, dtype=torch.long)
        if out.ndim != 1:
            raise ValueError("env_ids must be a 1D tensor")
        return out

    def _validate_actions(self, actions: torch.Tensor) -> torch.Tensor:
        out = actions.to(device=self.device, dtype=self.dtype)
        if tuple(out.shape) != self.action_shape:
            raise ValueError(f"expected action shape {self.action_shape}, got {tuple(out.shape)}")
        return out


def _mpe_simple_shader():
    return torch.mps.compile_shader(shader_source("mpe_simple_step.metal"))


__all__ = ["MPESimpleConfig", "MetalMPESimplePool"]
