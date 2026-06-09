"""Benchmark MetalEnvPool Atari preprocessing across common ALE games."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from metalenvpool import AtariPreprocessConfig, MetalAtariPreprocessor, memory_stats, synchronize

POPULAR_ATARI_ENVS = (
    "ALE/Breakout-v5",
    "ALE/Pong-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Seaquest-v5",
    "ALE/MsPacman-v5",
    "ALE/Qbert-v5",
    "ALE/BeamRider-v5",
    "ALE/Enduro-v5",
)


@dataclass(frozen=True)
class CapturedFrames:
    env_id: str
    action_space: str
    frame_a: torch.Tensor
    frame_b: torch.Tensor


def parse_envs(raw: str) -> tuple[str, ...]:
    envs = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not envs:
        raise ValueError("at least one env id is required")
    return envs


def capture_frames(env_id: str, cfg: AtariPreprocessConfig, seed: int) -> CapturedFrames:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)

    envs = [gym.make(env_id, render_mode="rgb_array") for _ in range(cfg.num_envs)]
    frames = []
    next_frames = []
    try:
        for i, env in enumerate(envs):
            env.reset(seed=seed + i)
            frame_a = env.render()
            action = env.action_space.sample()
            env.step(action)
            frame_b = env.render()
            frames.append(frame_a)
            next_frames.append(frame_b)
        action_space = str(envs[0].action_space)
    finally:
        for env in envs:
            env.close()

    return CapturedFrames(
        env_id=env_id,
        action_space=action_space,
        frame_a=torch.as_tensor(np.stack(frames, axis=0), dtype=torch.uint8),
        frame_b=torch.as_tensor(np.stack(next_frames, axis=0), dtype=torch.uint8),
    )


def benchmark_device(
    captured: CapturedFrames,
    cfg: AtariPreprocessConfig,
    *,
    device: str,
    iters: int,
    warmup: int,
    use_shader: bool = True,
) -> dict[str, object]:
    pre = MetalAtariPreprocessor(cfg, device=device, use_shader=use_shader)
    frame_a = captured.frame_a.to(pre.device)
    frame_b = captured.frame_b.to(pre.device)

    for i in range(warmup):
        pre.step(frame_a, frame_b, i % cfg.rollout_steps)
    synchronize(pre.device)

    t0 = time.perf_counter()
    for i in range(iters):
        pre.step(frame_a, frame_b, i % cfg.rollout_steps)
    synchronize(pre.device)
    seconds = time.perf_counter() - t0

    processed_obs = iters * cfg.num_envs
    raw_frames = processed_obs * 2
    out = {
        "device": str(pre.device),
        "using_shader": pre.using_shader,
        "seconds": seconds,
        "stacked_observations_per_second": processed_obs / seconds,
        "raw_frames_per_second": raw_frames / seconds,
        "rollout_obs_shape": list(pre.rollout.obs.shape),
        "rollout_obs_layout": pre.layout.describe(),
        "memory": memory_stats(pre.device).__dict__,
    }

    del pre, frame_a, frame_b
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def available_devices(raw: str) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not requested:
        raise ValueError("at least one device is required")
    if "auto" not in requested:
        return requested
    if torch.backends.mps.is_available():
        return ("mps", "cpu")
    return ("cpu",)


def make_markdown(results: Iterable[dict[str, object]]) -> str:
    rows = []
    for result in results:
        devices = result["devices"]
        if not isinstance(devices, dict):
            continue
        mps = devices.get("mps")
        cpu = devices.get("cpu")
        if not isinstance(mps, dict) or not isinstance(cpu, dict):
            continue
        mps_obs = float(mps["stacked_observations_per_second"])
        cpu_obs = float(cpu["stacked_observations_per_second"])
        rows.append(
            (
                str(result["env_id"]),
                str(result["action_space"]),
                mps_obs,
                cpu_obs,
                mps_obs / cpu_obs,
                float(mps["raw_frames_per_second"]),
                float(cpu["raw_frames_per_second"]),
            )
        )

    header = (
        "| Env | Action space | MPS stacked obs/s | CPU stacked obs/s | MPS/CPU | MPS raw frames/s | CPU raw frames/s |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for env_id, action_space, mps_obs, cpu_obs, speedup, mps_raw, cpu_raw in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    env_id,
                    action_space,
                    f"{mps_obs:,.0f}",
                    f"{cpu_obs:,.0f}",
                    f"{speedup:.2f}x",
                    f"{mps_raw:,.0f}",
                    f"{cpu_raw:,.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default=",".join(POPULAR_ATARI_ENVS))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    cfg = AtariPreprocessConfig(num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    env_ids = parse_envs(args.envs)
    devices = available_devices(args.devices)
    results = []

    for env_id in env_ids:
        captured = capture_frames(env_id, cfg, args.seed)
        entry: dict[str, object] = {
            "env_id": env_id,
            "action_space": captured.action_space,
            "devices": {},
        }
        for device in devices:
            device_result = benchmark_device(
                captured,
                cfg,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            )
            entry["devices"][device] = device_result  # type: ignore[index]
        results.append(entry)

    out = {
        "backend": "metal-atari-preprocess-suite",
        "source": "gymnasium-ale-rgb-frames",
        "num_envs": cfg.num_envs,
        "rollout_steps": cfg.rollout_steps,
        "iters": args.iters,
        "warmup": args.warmup,
        "note": "Timed path reuses captured ALE RGB frames; emulator stepping is not included.",
        "results": results,
    }
    if args.format == "markdown":
        print(make_markdown(results))
    else:
        print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
