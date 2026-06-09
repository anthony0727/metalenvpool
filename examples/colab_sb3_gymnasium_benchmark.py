"""Colab CLI benchmark for Stable-Baselines3 + Gymnasium.

Run from a local checkout with:

    colab run --gpu T4 examples/colab_sb3_gymnasium_benchmark.py --device auto
    colab run --tpu v5e1 examples/colab_sb3_gymnasium_benchmark.py --device auto

The TPU lane is intentionally honest: Stable-Baselines3 is PyTorch-based and is
not TPU-native. This script reports torch-xla availability, then falls back to
CPU unless a usable XLA device is present and explicitly requested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def maybe_install(packages: list[str]) -> None:
    missing = [module for module in ("gymnasium", "stable_baselines3") if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def command_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def accelerator_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [],
        "colab_tpu_addr": os.environ.get("COLAB_TPU_ADDR"),
        "tpu_name": os.environ.get("TPU_NAME"),
        "xrt_tpu_config": os.environ.get("XRT_TPU_CONFIG"),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm

        xla_device = xm.xla_device()
        info["torch_xla_available"] = True
        info["xla_device"] = str(xla_device)
    except Exception as exc:
        info["torch_xla_available"] = False
        info["torch_xla_error"] = f"{type(exc).__name__}: {exc}"
    return info


def resolve_device(requested: str, info: dict[str, Any]) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="InvertedDoublePendulum-v5")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("colab_sb3_gymnasium_result.json"))
    args = parser.parse_args()

    if not args.no_install:
        maybe_install(["stable-baselines3>=2.7.0", "gymnasium[mujoco]>=1.1.1"])

    import gymnasium as gym
    import numpy as np
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    info = accelerator_info()
    device = resolve_device(args.device, info)

    class Float32Observation(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("float32 observation wrapper only supports Box spaces")
            self.observation_space = gym.spaces.Box(
                low=env.observation_space.low.astype(np.float32),
                high=env.observation_space.high.astype(np.float32),
                shape=env.observation_space.shape,
                dtype=np.float32,
            )

        def observation(self, observation):
            return np.asarray(observation, dtype=np.float32)

    def make_env(rank: int):
        def thunk():
            env = gym.make(args.env_id)
            env.reset(seed=args.seed + rank)
            return Float32Observation(env)

        return thunk

    vec_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    env = vec_cls([make_env(i) for i in range(args.num_envs)])
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=0,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    seconds = time.perf_counter() - t0
    env.close()

    eval_mean = None
    eval_std = None
    if args.eval_episodes > 0:
        eval_env = make_env(10_000)()
        eval_mean, eval_std = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            warn=False,
        )
        eval_env.close()

    result = {
        "backend": "stable-baselines3-ppo",
        "library_versions": {
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
        "accelerator": info,
        "env_id": args.env_id,
        "vec_env": args.vec_env,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "requested_device": args.device,
        "resolved_device": device,
        "model_device": str(model.device),
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "eval_episodes": args.eval_episodes,
        "eval_mean_episode_return": eval_mean,
        "eval_std_episode_return": eval_std,
        "seconds": seconds,
        "env_steps_per_second": args.total_timesteps / seconds,
        "note": (
            "TPU runtimes are reported, but SB3/Gymnasium is not TPU-native; "
            "expect CPU fallback unless torch-xla interop works."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
