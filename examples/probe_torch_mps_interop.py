"""Probe whether Torch MPS tensors expose shared Metal buffer handles."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    root = Path(__file__).resolve().parents[1]
    ext = load(
        name="metalenvpool_torch_mps_interop_probe",
        sources=[str(root / "native" / "torch_mps_interop_probe.cpp")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )
    rollout = torch.empty((128, 16, 4, 84, 84), device="mps", dtype=torch.uint8)
    info = ext.mps_shared_buffer_info(rollout)
    info["shape"] = list(rollout.shape)
    info["dtype"] = str(rollout.dtype).replace("torch.", "")
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
