"""Rollout buffers laid out for learner consumption."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .device import resolve_device
from .layout import RolloutObsLayout


@dataclass(frozen=True)
class RolloutBufferConfig:
    """Static rollout-buffer shape."""

    num_steps: int
    num_envs: int
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...] = ()
    obs_dtype: torch.dtype = torch.float32
    action_dtype: torch.dtype = torch.int64


class MetalRolloutBuffer:
    """Preallocated rollout buffer on one device.

    Layout is intentionally learner-friendly:

    - observations: ``[T, N, *obs_shape]``
    - actions: ``[T, N, *action_shape]``
    - rewards/dones/logprobs/values: ``[T, N]``
    """

    def __init__(self, cfg: RolloutBufferConfig, *, device: str | torch.device = "auto") -> None:
        if cfg.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if cfg.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg
        self.device = resolve_device(device)
        self.obs_layout = RolloutObsLayout(cfg.num_steps, cfg.num_envs, cfg.obs_shape, cfg.obs_dtype)
        self.obs = torch.empty(self.obs_layout.rollout_obs_shape, device=self.device, dtype=cfg.obs_dtype)
        self.actions = torch.empty(
            (cfg.num_steps, cfg.num_envs, *cfg.action_shape),
            device=self.device,
            dtype=cfg.action_dtype,
        )
        self.rewards = torch.empty((cfg.num_steps, cfg.num_envs), device=self.device, dtype=torch.float32)
        self.terminated = torch.empty((cfg.num_steps, cfg.num_envs), device=self.device, dtype=torch.bool)
        self.truncated = torch.empty((cfg.num_steps, cfg.num_envs), device=self.device, dtype=torch.bool)
        self.logprobs = torch.empty((cfg.num_steps, cfg.num_envs), device=self.device, dtype=torch.float32)
        self.values = torch.empty((cfg.num_steps, cfg.num_envs), device=self.device, dtype=torch.float32)
        self.obs_layout.validate_rollout_obs(self.obs)

    def zero_(self) -> None:
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.terminated.zero_()
        self.truncated.zero_()
        self.logprobs.zero_()
        self.values.zero_()

    def learner_obs_view(self) -> torch.Tensor:
        """Return observations as ``[T * N, *obs_shape]`` without copying."""

        return self.obs_layout.learner_view(self.obs)


__all__ = ["MetalRolloutBuffer", "RolloutBufferConfig", "RolloutObsLayout"]
