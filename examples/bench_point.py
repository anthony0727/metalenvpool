"""Benchmark the fused Metal point-mass pool."""

from __future__ import annotations

import argparse
import json
import time

from metalenvpool import MetalPointPool, PointConfig, memory_stats, step_with_autoreset, synchronize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-shader", action="store_true")
    parser.add_argument("--action-mode", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    pool = MetalPointPool(PointConfig(num_envs=args.num_envs), device=args.device, use_shader=not args.no_shader)
    pool.reset(seed=args.seed)
    actions = pool.zero_actions()

    for _ in range(args.warmup):
        if args.action_mode == "random":
            actions = pool.sample_random_actions()
        step_with_autoreset(pool, actions)
    synchronize(pool.device)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        if args.action_mode == "random":
            actions = pool.sample_random_actions()
        step_with_autoreset(pool, actions)
    synchronize(pool.device)
    dt = time.perf_counter() - t0

    stats = memory_stats(pool.device)
    out = {
        "device": str(pool.device),
        "using_shader": pool.using_shader,
        "autoreset": True,
        "action_mode": args.action_mode,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "env_steps": args.num_envs * args.steps,
        "seconds": dt,
        "env_steps_per_second": (args.num_envs * args.steps) / dt,
        "obs_shape": list(pool.obs_shape),
        "action_shape": list(pool.action_shape),
        "memory": stats.__dict__,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
