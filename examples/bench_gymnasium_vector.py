"""Benchmark standard Gymnasium vector environment stepping."""

from __future__ import annotations

import argparse
import json
import time

import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    envs = gym.make_vec(args.env_id, num_envs=args.num_envs, vectorization_mode=args.mode)
    obs, _ = envs.reset(seed=args.seed)
    for _ in range(args.warmup):
        obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())

    t0 = time.perf_counter()
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
    dt = time.perf_counter() - t0
    envs.close()

    out = {
        "backend": "gymnasium-vector",
        "env_id": args.env_id,
        "mode": args.mode,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "env_steps": args.num_envs * args.steps,
        "seconds": dt,
        "env_steps_per_second": (args.num_envs * args.steps) / dt,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
