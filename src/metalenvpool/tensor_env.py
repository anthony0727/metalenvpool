"""Fast tensor-native environment contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .types import StepResult

_TORCH_TO_NUMPY = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


@dataclass(frozen=True)
class TensorSpec:
    """Tensor shape and bounds for one env slot."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    low: float | int | bool
    high: float | int | bool

    def batched_shape(self, num_envs: int) -> tuple[int, ...]:
        return (num_envs, *self.shape)

    def gymnasium_space(self) -> gym.Space:
        """Return a Gymnasium metadata space for framework integration."""

        try:
            np_dtype = _TORCH_TO_NUMPY[self.dtype]
        except KeyError as exc:
            raise TypeError(f"no NumPy dtype mapping for {self.dtype}") from exc
        return gym.spaces.Box(low=self.low, high=self.high, shape=self.shape, dtype=np_dtype)


class TensorEnv(ABC):
    """High-throughput batched env API.

    The contract is intentionally stricter than Gymnasium:

    - every hot-path value is a tensor on ``device``
    - ``step`` advances all env slots in one call
    - ``reset_done`` performs device-side autoreset from a bool mask
    - callers avoid ``.cpu()``, ``.numpy()``, and ``.item()`` inside rollouts

    Gymnasium/SB3 wrappers can be built on top, but this class is the primary
    API for high-speed Apple Silicon training code.
    """

    num_envs: int
    device: torch.device
    single_observation_spec: TensorSpec
    single_action_spec: TensorSpec

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.single_observation_spec.batched_shape(self.num_envs)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self.single_action_spec.batched_shape(self.num_envs)

    @property
    def single_observation_space(self) -> gym.Space:
        return self.single_observation_spec.gymnasium_space()

    @property
    def single_action_space(self) -> gym.Space:
        return self.single_action_spec.gymnasium_space()

    @property
    def observation_space(self) -> gym.Space:
        return gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

    @property
    def action_space(self) -> gym.Space:
        return gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> torch.Tensor:
        """Reset all env slots and return batched observations on ``device``."""

    @abstractmethod
    def reset_done(self, done: torch.Tensor) -> torch.Tensor:
        """Reset env slots selected by a device bool mask."""

    @abstractmethod
    def step(self, actions: torch.Tensor) -> StepResult:
        """Advance all env slots using batched tensor actions."""

    @abstractmethod
    def sample_random_actions(self) -> torch.Tensor:
        """Sample batched random actions on ``device``."""

    @abstractmethod
    def zero_actions(self) -> torch.Tensor:
        """Return batched zero actions on ``device``."""

    def send(self, actions: torch.Tensor) -> None:
        self._pending_actions = actions

    def recv(self) -> StepResult:
        try:
            actions = self._pending_actions
        except AttributeError as exc:
            raise RuntimeError("recv called before send") from exc
        del self._pending_actions
        return self.step(actions)

    def close(self) -> None:
        """Release resources if a backend owns external handles."""
        return None


def check_tensor_env(env: TensorEnv, *, seed: int = 0) -> dict[str, Any]:
    """Lightweight runtime check for custom ``TensorEnv`` implementations."""

    obs = env.reset(seed=seed)
    if tuple(obs.shape) != env.obs_shape:
        raise AssertionError(f"reset obs shape {tuple(obs.shape)} != {env.obs_shape}")
    if obs.device != env.device:
        raise AssertionError(f"reset obs device {obs.device} != {env.device}")
    if obs.dtype != env.single_observation_spec.dtype:
        raise AssertionError(f"reset obs dtype {obs.dtype} != {env.single_observation_spec.dtype}")

    actions = env.zero_actions()
    if tuple(actions.shape) != env.action_shape:
        raise AssertionError(f"action shape {tuple(actions.shape)} != {env.action_shape}")
    if actions.device != env.device:
        raise AssertionError(f"action device {actions.device} != {env.device}")
    if actions.dtype != env.single_action_spec.dtype:
        raise AssertionError(f"action dtype {actions.dtype} != {env.single_action_spec.dtype}")

    step = env.step(actions)
    if tuple(step.obs.shape) != env.obs_shape:
        raise AssertionError(f"step obs shape {tuple(step.obs.shape)} != {env.obs_shape}")
    if tuple(step.reward.shape) != (env.num_envs,):
        raise AssertionError(f"reward shape {tuple(step.reward.shape)} != {(env.num_envs,)}")
    if tuple(step.terminated.shape) != (env.num_envs,):
        raise AssertionError(f"terminated shape {tuple(step.terminated.shape)} != {(env.num_envs,)}")
    if tuple(step.truncated.shape) != (env.num_envs,):
        raise AssertionError(f"truncated shape {tuple(step.truncated.shape)} != {(env.num_envs,)}")
    done = step.terminated | step.truncated
    reset_obs = env.reset_done(done)
    if tuple(reset_obs.shape) != env.obs_shape:
        raise AssertionError(f"reset_done obs shape {tuple(reset_obs.shape)} != {env.obs_shape}")
    return {
        "num_envs": env.num_envs,
        "device": str(env.device),
        "obs_shape": env.obs_shape,
        "action_shape": env.action_shape,
    }


__all__ = ["TensorEnv", "TensorSpec", "check_tensor_env"]
