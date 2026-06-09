"""Atari frame preprocessing into an Apple-SoC rollout layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .device import resolve_device
from .native import shader_source
from .rollout import MetalRolloutBuffer, RolloutBufferConfig


@dataclass(frozen=True)
class AtariPreprocessConfig:
    """Static Atari preprocessing shape."""

    num_envs: int = 16
    rollout_steps: int = 128
    in_height: int = 210
    in_width: int = 160
    out_height: int = 84
    out_width: int = 84
    frame_stack: int = 4

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.frame_stack, self.out_height, self.out_width)


class MetalAtariPreprocessor:
    """Fused max-pool, grayscale, resize, frame-stack, rollout write.

    The active backend uses PyTorch's runtime Metal shader compiler on MPS.
    The API is shaped so a native metal-cpp backend can replace it without
    changing the benchmark or learner-facing layout.
    """

    def __init__(
        self,
        cfg: AtariPreprocessConfig | None = None,
        *,
        device: str | torch.device = "auto",
        use_shader: bool = True,
    ) -> None:
        self.cfg = cfg or AtariPreprocessConfig()
        if self.cfg.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.cfg.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self.device = resolve_device(device)
        self.rollout = MetalRolloutBuffer(
            RolloutBufferConfig(
                num_steps=self.cfg.rollout_steps,
                num_envs=self.cfg.num_envs,
                obs_shape=self.cfg.obs_shape,
                obs_dtype=torch.uint8,
            ),
            device=self.device,
        )
        self.layout = self.rollout.obs_layout
        self.obs = torch.zeros(self.layout.current_obs_shape, device=self.device, dtype=torch.uint8)
        self.layout.validate_current_obs(self.obs, name="obs")
        self._shader = _atari_shader() if use_shader and self.device.type == "mps" else None

    @property
    def using_shader(self) -> bool:
        return self._shader is not None

    def reset(self) -> None:
        self.obs.zero_()

    def step(self, frame_a: torch.Tensor, frame_b: torch.Tensor, step_index: int) -> torch.Tensor:
        """Process two RGB frames and write stacked observation to rollout.

        ``frame_a`` and ``frame_b`` are raw Atari RGB frames with shape
        ``[N, 210, 160, 3]`` by default. The max of the two frames implements
        the standard Atari flicker-reduction max-pool stage.
        """

        if not 0 <= step_index < self.cfg.rollout_steps:
            raise ValueError(f"step_index must be in [0, {self.cfg.rollout_steps})")
        a = self._validate_frame(frame_a)
        b = self._validate_frame(frame_b)
        if self._shader is None:
            self._reference_step(a, b, step_index)
        else:
            cfg = self.cfg
            self._shader.atari_preprocess_write(
                a,
                b,
                self.obs,
                self.rollout.obs,
                cfg.num_envs,
                cfg.in_height,
                cfg.in_width,
                cfg.out_height,
                cfg.out_width,
                cfg.frame_stack,
                step_index,
            )
        return self.rollout.obs[step_index]

    def _validate_frame(self, frame: torch.Tensor) -> torch.Tensor:
        out = frame.to(device=self.device, dtype=torch.uint8)
        expected = (self.cfg.num_envs, self.cfg.in_height, self.cfg.in_width, 3)
        if tuple(out.shape) != expected:
            raise ValueError(f"expected frame shape {expected}, got {tuple(out.shape)}")
        return out.contiguous()

    def _reference_step(self, frame_a: torch.Tensor, frame_b: torch.Tensor, step_index: int) -> None:
        cfg = self.cfg
        pooled = torch.maximum(frame_a, frame_b)
        y_idx = (torch.arange(cfg.out_height, device=self.device) * cfg.in_height) // cfg.out_height
        x_idx = (torch.arange(cfg.out_width, device=self.device) * cfg.in_width) // cfg.out_width
        sampled = pooled[:, y_idx[:, None], x_idx[None, :], :].to(torch.int32)
        gray = ((77 * sampled[..., 0]) + (150 * sampled[..., 1]) + (29 * sampled[..., 2])) >> 8
        if cfg.frame_stack > 1:
            self.obs[:, :-1].copy_(self.obs[:, 1:].clone())
        self.obs[:, -1].copy_(gray.to(torch.uint8))
        self.rollout.obs[step_index].copy_(self.obs)


def _atari_shader():
    return torch.mps.compile_shader(shader_source("atari_preprocess.metal"))


__all__ = ["AtariPreprocessConfig", "MetalAtariPreprocessor"]
