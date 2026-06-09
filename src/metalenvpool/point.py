"""Fused Metal point-mass environment."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .device import resolve_device, seed_device
from .native import shader_source
from .tensor_env import TensorEnv, TensorSpec
from .types import StepResult


@dataclass(frozen=True)
class PointConfig:
    """Static config for a tiny continuous-control task."""

    num_envs: int = 65536
    world_size: float = 10.0
    dt: float = 0.05
    max_steps: int = 200
    max_speed: float = 4.0
    max_accel: float = 8.0
    drag: float = 0.03
    success_radius: float = 0.15


class MetalPointPool(TensorEnv):
    """Point-mass task with a fused Metal shader step on MPS.

    State is ``[x, y, vx, vy]`` and target is ``[tx, ty]`` per env. Actions are
    accelerations ``[ax, ay]``. The MPS path launches one custom Metal kernel
    per step, which is the shape of optimization this repo needs for EnvPool-
    style throughput on Apple GPUs.
    """

    def __init__(
        self,
        cfg: PointConfig | None = None,
        *,
        device: str | torch.device = "auto",
        use_shader: bool = True,
    ) -> None:
        self.cfg = cfg or PointConfig()
        if self.cfg.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = self.cfg.num_envs
        self.device = resolve_device(device)
        self.dtype = torch.float32
        self.single_observation_spec = TensorSpec(
            shape=(6,),
            dtype=self.dtype,
            low=-float("inf"),
            high=float("inf"),
        )
        self.single_action_spec = TensorSpec(
            shape=(2,),
            dtype=self.dtype,
            low=-self.cfg.max_accel,
            high=self.cfg.max_accel,
        )
        self.state = torch.empty((self.cfg.num_envs, 4), device=self.device, dtype=self.dtype)
        self.target = torch.empty((self.cfg.num_envs, 2), device=self.device, dtype=self.dtype)
        self.reward = torch.empty((self.cfg.num_envs,), device=self.device, dtype=self.dtype)
        self.terminated = torch.empty((self.cfg.num_envs,), device=self.device, dtype=torch.bool)
        self.truncated = torch.empty((self.cfg.num_envs,), device=self.device, dtype=torch.bool)
        self.steps = torch.empty((self.cfg.num_envs,), device=self.device, dtype=torch.int32)
        self._pending_actions: torch.Tensor | None = None
        self._shader = _point_shader() if use_shader and self.device.type == "mps" else None
        self.reset()

    @property
    def action_shape(self) -> tuple[int, int]:
        return (self.num_envs, 2)

    @property
    def obs_shape(self) -> tuple[int, int]:
        return (self.num_envs, 6)

    @property
    def using_shader(self) -> bool:
        return self._shader is not None

    def reset(self, *, seed: int | None = None, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if seed is not None:
            seed_device(seed, self.device)
        ids = self._env_ids(env_ids)
        n = int(ids.numel())
        self.state[ids, :2] = self._rand((n, 2)) * self.cfg.world_size
        self.state[ids, 2:] = 0
        self.target[ids] = self._rand((n, 2)) * self.cfg.world_size
        self.reward[ids] = 0
        self.terminated[ids] = False
        self.truncated[ids] = False
        self.steps[ids] = 0
        return self.observe()

    def reset_done(self, done: torch.Tensor) -> torch.Tensor:
        """Reset slots selected by a device-side done mask.

        This is the hot-path autoreset primitive for tensor learners. It avoids
        querying ``done.any()`` from Python, which would synchronize MPS work.
        """

        mask = done.to(device=self.device, dtype=torch.bool)
        if tuple(mask.shape) != (self.cfg.num_envs,):
            raise ValueError(f"expected done shape {(self.cfg.num_envs,)}, got {tuple(mask.shape)}")
        mask2 = mask.unsqueeze(1)
        new_pos = self._rand((self.cfg.num_envs, 2)) * self.cfg.world_size
        new_target = self._rand((self.cfg.num_envs, 2)) * self.cfg.world_size
        zeros2 = torch.zeros((self.cfg.num_envs, 2), device=self.device, dtype=self.dtype)
        self.state[:, :2] = torch.where(mask2, new_pos, self.state[:, :2])
        self.state[:, 2:] = torch.where(mask2, zeros2, self.state[:, 2:])
        self.target[:] = torch.where(mask2, new_target, self.target)
        self.terminated[:] = torch.where(mask, torch.zeros_like(self.terminated), self.terminated)
        self.truncated[:] = torch.where(mask, torch.zeros_like(self.truncated), self.truncated)
        self.steps[:] = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        return self.observe()

    def observe(self) -> torch.Tensor:
        return torch.cat([self.state, self.target], dim=1)

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
            self._shader.point_step(
                self.state,
                actions,
                self.target,
                self.reward,
                self.terminated,
                self.truncated,
                self.steps,
                cfg.num_envs,
                cfg.dt,
                cfg.max_speed,
                cfg.max_accel,
                cfg.drag,
                cfg.world_size,
                cfg.success_radius,
                cfg.max_steps,
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
        return ((self._rand(self.action_shape) * 2.0) - 1.0) * self.cfg.max_accel

    def zero_actions(self) -> torch.Tensor:
        return torch.zeros(self.action_shape, device=self.device, dtype=self.dtype)

    def _torch_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        active = ~(self.terminated | self.truncated)
        accel = actions.clamp(-cfg.max_accel, cfg.max_accel)
        vel = (self.state[:, 2:] * (1.0 - cfg.drag)) + (accel * cfg.dt)
        speed = torch.linalg.vector_norm(vel, dim=1, keepdim=True).clamp_min(1e-8)
        vel = vel * torch.clamp(cfg.max_speed / speed, max=1.0)
        pos = self.state[:, :2] + (vel * cfg.dt)
        below = pos < 0.0
        above = pos > cfg.world_size
        bounced = below | above
        pos = pos.clamp(0.0, cfg.world_size)
        vel = torch.where(bounced, -0.5 * vel, vel)
        self.state[:, :2] = torch.where(active.unsqueeze(1), pos, self.state[:, :2])
        self.state[:, 2:] = torch.where(active.unsqueeze(1), vel, self.state[:, 2:])
        self.steps += active.to(torch.int32)

        delta = self.target - self.state[:, :2]
        dist = torch.linalg.vector_norm(delta, dim=1)
        hit = (dist <= cfg.success_radius) & active
        self.terminated[:] = self.terminated | hit
        self.truncated[:] = self.truncated | ((self.steps >= cfg.max_steps) & ~self.terminated)
        self.reward[:] = torch.where(active, -dist - 0.001 + hit.to(self.dtype), torch.zeros_like(dist))

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, device=self.device, dtype=self.dtype)

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


def _point_shader():
    return torch.mps.compile_shader(shader_source("point_step.metal"))


__all__ = ["MetalPointPool", "PointConfig"]
