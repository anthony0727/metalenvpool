"""Benchmark Stable-Baselines3 PPO training wall-clock throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=25000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/sb3_ppo"))
    args = parser.parse_args()

    vec_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    env = make_vec_env(args.env_id, n_envs=args.num_envs, seed=args.seed, vec_env_cls=vec_cls)
    model = PPO(
        "MlpPolicy",
        env,
        device=args.device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=0,
        tensorboard_log=str(args.log_dir),
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    dt = time.perf_counter() - t0
    env.close()

    out = {
        "backend": "stable-baselines3-ppo",
        "env_id": args.env_id,
        "vec_env": args.vec_env,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "device": args.device,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "seconds": dt,
        "env_steps_per_second": args.total_timesteps / dt,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
