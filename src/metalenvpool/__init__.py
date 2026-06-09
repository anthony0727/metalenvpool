"""EnvPool-style batched environments for Apple Metal/MPS."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AtariPreprocessConfig",
    "DeviceStats",
    "MetalAtariPreprocessor",
    "MetalPointPool",
    "MetalRolloutBuffer",
    "PointConfig",
    "RolloutBufferConfig",
    "RolloutObsLayout",
    "StepResult",
    "TensorEnv",
    "TensorSpec",
    "check_tensor_env",
    "memory_stats",
    "resolve_device",
    "synchronize",
]

_EXPORTS = {
    "AtariPreprocessConfig": ("metalenvpool.atari", "AtariPreprocessConfig"),
    "DeviceStats": ("metalenvpool.device", "DeviceStats"),
    "MetalAtariPreprocessor": ("metalenvpool.atari", "MetalAtariPreprocessor"),
    "MetalPointPool": ("metalenvpool.point", "MetalPointPool"),
    "MetalRolloutBuffer": ("metalenvpool.rollout", "MetalRolloutBuffer"),
    "PointConfig": ("metalenvpool.point", "PointConfig"),
    "RolloutBufferConfig": ("metalenvpool.rollout", "RolloutBufferConfig"),
    "RolloutObsLayout": ("metalenvpool.layout", "RolloutObsLayout"),
    "StepResult": ("metalenvpool.types", "StepResult"),
    "TensorEnv": ("metalenvpool.tensor_env", "TensorEnv"),
    "TensorSpec": ("metalenvpool.tensor_env", "TensorSpec"),
    "check_tensor_env": ("metalenvpool.tensor_env", "check_tensor_env"),
    "memory_stats": ("metalenvpool.device", "memory_stats"),
    "resolve_device": ("metalenvpool.device", "resolve_device"),
    "synchronize": ("metalenvpool.device", "synchronize"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(module_name), attr_name)
