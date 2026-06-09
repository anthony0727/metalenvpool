"""Memory-layout contracts for env buffers and learner rollout tensors."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import torch


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_index(name: str, value: int, limit: int) -> None:
    if not 0 <= value < limit:
        raise IndexError(f"{name}={value} outside [0, {limit})")


@dataclass(frozen=True)
class RolloutObsLayout:
    """Contiguous learner ABI for rollout observations.

    The hot layout is ``[T, N, *obs_shape]``. For Atari this becomes
    ``[T, N, C, H, W]`` uint8, so PPO/DQN can flatten to
    ``[T * N, C, H, W]`` as a view and the shader can write exactly where the
    learner will read.
    """

    num_steps: int
    num_envs: int
    obs_shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        _validate_positive("num_steps", self.num_steps)
        _validate_positive("num_envs", self.num_envs)
        if not self.obs_shape:
            raise ValueError("obs_shape must not be empty")
        for i, dim in enumerate(self.obs_shape):
            _validate_positive(f"obs_shape[{i}]", dim)

    @property
    def obs_elements(self) -> int:
        return prod(self.obs_shape)

    @property
    def rollout_obs_shape(self) -> tuple[int, ...]:
        return (self.num_steps, self.num_envs, *self.obs_shape)

    @property
    def current_obs_shape(self) -> tuple[int, ...]:
        return (self.num_envs, *self.obs_shape)

    @property
    def learner_batch_shape(self) -> tuple[int, ...]:
        return (self.num_steps * self.num_envs, *self.obs_shape)

    @property
    def obs_strides(self) -> tuple[int, ...]:
        return _row_major_strides(self.obs_shape)

    @property
    def current_obs_strides(self) -> tuple[int, ...]:
        return (self.obs_elements, *self.obs_strides)

    @property
    def rollout_obs_strides(self) -> tuple[int, ...]:
        step_stride = self.num_envs * self.obs_elements
        return (step_stride, self.obs_elements, *self.obs_strides)

    @property
    def rollout_obs_elements(self) -> int:
        return self.num_steps * self.num_envs * self.obs_elements

    @property
    def element_size_bytes(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def rollout_obs_bytes(self) -> int:
        return self.rollout_obs_elements * self.element_size_bytes

    def learner_batch_index(self, step: int, env: int) -> int:
        _validate_index("step", step, self.num_steps)
        _validate_index("env", env, self.num_envs)
        return (step * self.num_envs) + env

    def current_obs_offset(self, env: int, *obs_index: int) -> int:
        _validate_index("env", env, self.num_envs)
        if len(obs_index) != len(self.obs_shape):
            raise ValueError(f"expected {len(self.obs_shape)} obs indices, got {len(obs_index)}")
        offset = env * self.current_obs_strides[0]
        for axis, idx in enumerate(obs_index):
            _validate_index(f"obs_index[{axis}]", idx, self.obs_shape[axis])
            offset += idx * self.obs_strides[axis]
        return offset

    def rollout_obs_offset(self, step: int, env: int, *obs_index: int) -> int:
        _validate_index("step", step, self.num_steps)
        return (step * self.rollout_obs_strides[0]) + self.current_obs_offset(env, *obs_index)

    def validate_current_obs(self, tensor: torch.Tensor, *, name: str = "current_obs") -> None:
        self._validate_tensor(tensor, self.current_obs_shape, name=name)

    def validate_rollout_obs(self, tensor: torch.Tensor, *, name: str = "rollout_obs") -> None:
        self._validate_tensor(tensor, self.rollout_obs_shape, name=name)

    def learner_view(self, rollout_obs: torch.Tensor) -> torch.Tensor:
        self.validate_rollout_obs(rollout_obs)
        return rollout_obs.view(self.learner_batch_shape)

    def describe(self) -> dict[str, Any]:
        return {
            "order": "time_env_observation",
            "rollout_obs_shape": list(self.rollout_obs_shape),
            "rollout_obs_strides": list(self.rollout_obs_strides),
            "current_obs_shape": list(self.current_obs_shape),
            "current_obs_strides": list(self.current_obs_strides),
            "learner_batch_shape": list(self.learner_batch_shape),
            "dtype": str(self.dtype).replace("torch.", ""),
            "rollout_obs_bytes": self.rollout_obs_bytes,
        }

    def _validate_tensor(self, tensor: torch.Tensor, shape: tuple[int, ...], *, name: str) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != self.dtype:
            raise TypeError(f"{name} dtype must be {self.dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


__all__ = ["RolloutObsLayout"]
