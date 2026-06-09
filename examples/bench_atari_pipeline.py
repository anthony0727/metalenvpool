"""Benchmark Atari preprocessing into a MetalEnvPool rollout buffer."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from metalenvpool import AtariPreprocessConfig, MetalAtariPreprocessor, memory_stats, synchronize


def synthetic_frames(cfg: AtariPreprocessConfig, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randint(0, 256, (cfg.num_envs, cfg.in_height, cfg.in_width, 3), device=device, dtype=torch.uint8)
    b = torch.randint(0, 256, (cfg.num_envs, cfg.in_height, cfg.in_width, 3), device=device, dtype=torch.uint8)
    return a, b


def gym_frames(env_id: str, cfg: AtariPreprocessConfig) -> tuple[torch.Tensor, torch.Tensor]:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)

    envs = [gym.make(env_id, render_mode="rgb_array") for _ in range(cfg.num_envs)]
    frames = []
    next_frames = []
    for i, env in enumerate(envs):
        env.reset(seed=7 + i)
        frame_a = env.render()
        action = env.action_space.sample()
        env.step(action)
        frame_b = env.render()
        env.close()
        frames.append(frame_a)
        next_frames.append(frame_b)
    a = torch.as_tensor(np.stack(frames, axis=0), dtype=torch.uint8)
    b = torch.as_tensor(np.stack(next_frames, axis=0), dtype=torch.uint8)
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source", choices=["synthetic", "gym"], default="synthetic")
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--no-shader", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = AtariPreprocessConfig(num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    pre = MetalAtariPreprocessor(cfg, device=args.device, use_shader=not args.no_shader)
    if args.source == "gym":
        try:
            frame_a, frame_b = gym_frames(args.env_id, cfg)
        except Exception as exc:
            out = {
                "backend": "metal-atari-preprocess",
                "status": "blocked",
                "source": "gym",
                "env_id": args.env_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": (
                    "Synthetic Atari-frame preprocessing remains runnable; real ALE/Breakout needs a local ROM install."
                ),
            }
            print(json.dumps(out, indent=2, sort_keys=True))
            raise SystemExit(2) from exc
        frame_a = frame_a.to(pre.device)
        frame_b = frame_b.to(pre.device)
    else:
        frame_a, frame_b = synthetic_frames(cfg, pre.device, args.seed)

    for i in range(args.warmup):
        pre.step(frame_a, frame_b, i % cfg.rollout_steps)
    synchronize(pre.device)

    t0 = time.perf_counter()
    for i in range(args.iters):
        pre.step(frame_a, frame_b, i % cfg.rollout_steps)
    synchronize(pre.device)
    dt = time.perf_counter() - t0

    raw_frames = args.iters * cfg.num_envs * 2
    processed_obs = args.iters * cfg.num_envs
    out = {
        "backend": "metal-atari-preprocess",
        "device": str(pre.device),
        "using_shader": pre.using_shader,
        "source": args.source,
        "env_id": args.env_id if args.source == "gym" else None,
        "seed": args.seed,
        "num_envs": cfg.num_envs,
        "rollout_steps": cfg.rollout_steps,
        "iters": args.iters,
        "seconds": dt,
        "raw_frames_per_second": raw_frames / dt,
        "stacked_observations_per_second": processed_obs / dt,
        "rollout_obs_shape": list(pre.rollout.obs.shape),
        "rollout_obs_layout": pre.layout.describe(),
        "obs_dtype": str(pre.rollout.obs.dtype).replace("torch.", ""),
        "memory": memory_stats(pre.device).__dict__,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
