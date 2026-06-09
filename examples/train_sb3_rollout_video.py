"""Train SB3 PPO and export a deterministic rollout video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def make_render_vec_env(env_id: str, seed: int) -> DummyVecEnv:
    def make_env():
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        return env

    return DummyVecEnv([make_env])


def make_eval_env(env_id: str, seed: int):
    def make_env():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([make_env])


def load_eval_normalizer(vecnormalize_path: Path | None, env):
    if vecnormalize_path is None:
        return env
    normalized = VecNormalize.load(str(vecnormalize_path), env)
    normalized.training = False
    normalized.norm_reward = False
    return normalized


def evaluate(model: PPO, env_id: str, episodes: int, seed: int, vecnormalize_path: Path | None) -> list[float]:
    returns: list[float] = []
    for idx in range(episodes):
        env = load_eval_normalizer(vecnormalize_path, make_eval_env(env_id, seed + idx))
        obs = env.reset()
        episode_return = 0.0
        done = np.array([False])
        while not bool(done[0]):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            episode_return += float(reward[0])
        returns.append(episode_return)
        env.close()
    return returns


def record_rollout(
    model: PPO,
    env_id: str,
    video_path: Path,
    *,
    seed: int,
    max_steps: int,
    fps: int,
    vecnormalize_path: Path | None,
) -> dict[str, float | int | str]:
    env = load_eval_normalizer(vecnormalize_path, make_render_vec_env(env_id, seed))
    obs = env.reset()
    first_frame = env.render(mode="rgb_array")
    frames = [first_frame]
    episode_return = 0.0
    steps = 0
    done = np.array([False])
    while not bool(done[0]) and steps < max_steps:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _info = env.step(action)
        episode_return += float(reward[0])
        frames.append(env.render(mode="rgb_array"))
        steps += 1
    env.close()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=16)
    return {
        "video_path": str(video_path),
        "video_frames": len(frames),
        "video_steps": steps,
        "video_return": episode_return,
        "video_fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="InvertedDoublePendulum-v5")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-rollout-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/ppo_rollouts"))
    args = parser.parse_args()

    vec_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    env = make_vec_env(args.env_id, n_envs=args.num_envs, seed=args.seed, vec_env_cls=vec_cls)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        device=args.device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        verbose=0,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    train_seconds = time.perf_counter() - t0
    env.close()

    safe_env = args.env_id.replace("/", "_").replace(":", "_")
    model_path = args.out_dir / f"{safe_env}_ppo"
    model.save(model_path)
    vecnormalize_path = args.out_dir / f"{safe_env}_vecnormalize.pkl" if args.normalize else None
    if vecnormalize_path is not None:
        model.get_vec_normalize_env().save(str(vecnormalize_path))
    returns = evaluate(model, args.env_id, args.eval_episodes, args.seed + 10_000, vecnormalize_path)
    video_path = args.out_dir / f"{safe_env}_ppo_rollout.mp4"
    video = record_rollout(
        model,
        args.env_id,
        video_path,
        seed=args.seed + 20_000,
        max_steps=args.max_rollout_steps,
        fps=args.fps,
        vecnormalize_path=vecnormalize_path,
    )

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
        "seconds": train_seconds,
        "env_steps_per_second": args.total_timesteps / train_seconds,
        "eval_episodes": args.eval_episodes,
        "eval_returns": returns,
        "eval_mean_return": float(np.mean(returns)),
        "eval_max_return": float(np.max(returns)),
        "model_path": f"{model_path}.zip",
        "normalize": args.normalize,
        "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path is not None else None,
        **video,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
