"""Shared result types."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StepResult:
    """Tensor step result."""

    obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, torch.Tensor]
