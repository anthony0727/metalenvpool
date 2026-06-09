"""Device helpers for Apple MPS and CPU fallback."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceStats:
    """Best-effort device memory counters."""

    device: str
    current_allocated_bytes: int | None
    driver_allocated_bytes: int | None
    recommended_max_bytes: int | None


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""

    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = torch.device(device)
    if out.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested device='mps', but PyTorch MPS is not available")
    return out


def synchronize(device: str | torch.device) -> None:
    """Synchronize device work for honest timing."""

    resolved = resolve_device(device)
    if resolved.type == "mps":
        torch.mps.synchronize()


def seed_device(seed: int, device: torch.device) -> None:
    """Seed torch RNGs used by this backend."""

    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def memory_stats(device: str | torch.device = "auto") -> DeviceStats:
    """Return MPS memory stats where available."""

    resolved = resolve_device(device)
    if resolved.type != "mps":
        return DeviceStats(str(resolved), None, None, None)
    return DeviceStats(
        device=str(resolved),
        current_allocated_bytes=int(torch.mps.current_allocated_memory()),
        driver_allocated_bytes=int(torch.mps.driver_allocated_memory()),
        recommended_max_bytes=int(torch.mps.recommended_max_memory()),
    )
