"""Small-CNN Breakout training-loop comparison.

This benchmark is intentionally not an EnvPool-paper reproduction. It targets a
smaller systems claim: for Atari-style deep RL, compare EnvPool's full env path
against a MetalEnvPool path that uses Gymnasium/ALE frames plus fused Metal
preprocessing and a tiny CNN learner.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from metalenvpool import AtariPreprocessConfig, MetalAtariPreprocessor, resolve_device, synchronize


class TinyAtariActorCritic(nn.Module):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        h = self.net(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["envpool", "metalenvpool"], required=True)
    parser.add_argument("--env-id", default="Breakout-v5")
    parser.add_argument("--gym-env-id", default="ALE/Breakout-v5")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.backend == "envpool":
        result = train_envpool(args)
    else:
        result = train_metalenvpool(args)

    emit_result(result, args.output_json)


def train_envpool(args: argparse.Namespace) -> dict[str, Any]:
    import envpool

    torch.manual_seed(args.seed)
    env = envpool.make(args.env_id, env_type="gymnasium", num_envs=args.num_envs)
    obs_np, _info = env.reset()
    device = resolve_device(args.device)
    agent, optimizer = build_agent_optimizer(env.spec.action_space.n, device, args.learning_rate)
    num_updates = max(1, args.total_timesteps // (args.num_envs * args.rollout_steps))
    measured_steps = num_updates * args.num_envs * args.rollout_steps

    start = time.perf_counter()
    total_reward = 0.0
    last_loss = torch.zeros((), device=device)
    for _ in range(num_updates):
        logprobs = []
        values = []
        rewards = []
        entropies = []
        for _step in range(args.rollout_steps):
            obs = torch.as_tensor(obs_np, device=device, dtype=torch.uint8)
            logits, value = agent(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, terminated, truncated, _info = env.step(action.cpu().numpy().astype(np.int64))
            logprobs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.as_tensor(reward, device=device, dtype=torch.float32))
            entropies.append(dist.entropy())
            total_reward += float(np.asarray(reward).sum())
            obs_np = next_obs

        loss = actor_critic_loss(logprobs, values, rewards, entropies, args.value_coef, args.entropy_coef)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()

    synchronize(device)
    seconds = time.perf_counter() - start
    env.close()
    return {
        "backend": "envpool-breakout-small-cnn",
        "env_id": args.env_id,
        "device": str(device),
        "full_env_execution": True,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "num_updates": num_updates,
        "total_timesteps": measured_steps,
        "seconds": seconds,
        "env_steps_per_second": measured_steps / seconds,
        "total_reward": total_reward,
        "last_loss": float(last_loss.cpu()),
        "note": "EnvPool performs full Atari env execution and returns stacked 84x84 observations.",
    }


def train_metalenvpool(args: argparse.Namespace) -> dict[str, Any]:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    cfg = AtariPreprocessConfig(num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    pre = MetalAtariPreprocessor(cfg, device=device, use_shader=True)
    envs = [gym.make(args.gym_env_id, render_mode="rgb_array") for _ in range(args.num_envs)]
    try:
        frame_a = []
        frame_b = []
        for i, env in enumerate(envs):
            env.reset(seed=args.seed + i)
            frame = env.render()
            frame_a.append(frame)
            frame_b.append(frame)
        prev = np.stack(frame_a, axis=0).astype(np.uint8)
        curr = np.stack(frame_b, axis=0).astype(np.uint8)

        agent, optimizer = build_agent_optimizer(envs[0].action_space.n, device, args.learning_rate)
        num_updates = max(1, args.total_timesteps // (args.num_envs * args.rollout_steps))
        measured_steps = num_updates * args.num_envs * args.rollout_steps

        start = time.perf_counter()
        total_reward = 0.0
        last_loss = torch.zeros((), device=device)
        for _ in range(num_updates):
            logprobs = []
            values = []
            rewards = []
            entropies = []
            for step in range(args.rollout_steps):
                obs = pre.step(
                    torch.as_tensor(prev, device=device, dtype=torch.uint8),
                    torch.as_tensor(curr, device=device, dtype=torch.uint8),
                    step,
                )
                logits, value = agent(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                actions_np = action.cpu().numpy()
                reward_np = np.zeros((args.num_envs,), dtype=np.float32)
                for i, env in enumerate(envs):
                    prev[i] = curr[i]
                    next_frame, reward, terminated, truncated, _info = env.step(int(actions_np[i]))
                    if terminated or truncated:
                        next_frame, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    curr[i] = next_frame
                    reward_np[i] = reward
                logprobs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(torch.as_tensor(reward_np, device=device, dtype=torch.float32))
                entropies.append(dist.entropy())
                total_reward += float(reward_np.sum())

            loss = actor_critic_loss(logprobs, values, rewards, entropies, args.value_coef, args.entropy_coef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()

        synchronize(device)
        seconds = time.perf_counter() - start
    finally:
        for env in envs:
            env.close()
    return {
        "backend": "metalenvpool-breakout-small-cnn",
        "env_id": args.gym_env_id,
        "device": str(device),
        "using_shader": pre.using_shader,
        "full_env_execution": True,
        "metalenvpool_component": "Atari preprocessing and learner-ready rollout write",
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "num_updates": num_updates,
        "total_timesteps": measured_steps,
        "seconds": seconds,
        "env_steps_per_second": measured_steps / seconds,
        "total_reward": total_reward,
        "last_loss": float(last_loss.cpu()),
        "note": "Gymnasium/ALE performs env execution; MetalEnvPool performs fused preprocessing/write on MPS.",
    }


def actor_critic_loss(
    logprobs: list[torch.Tensor],
    values: list[torch.Tensor],
    rewards: list[torch.Tensor],
    entropies: list[torch.Tensor],
    value_coef: float,
    entropy_coef: float,
) -> torch.Tensor:
    r = torch.stack(rewards)
    lp = torch.stack(logprobs)
    v = torch.stack(values)
    entropy = torch.stack(entropies)
    returns = discounted_returns(r, gamma=0.99)
    adv = returns - v.detach()
    policy_loss = -(lp * adv).mean()
    value_loss = 0.5 * (returns - v).square().mean()
    entropy_loss = entropy.mean()
    return policy_loss + value_coef * value_loss - entropy_coef * entropy_loss


def build_agent_optimizer(
    num_actions: int,
    device: torch.device,
    learning_rate: float,
) -> tuple[TinyAtariActorCritic, torch.optim.Optimizer]:
    agent = TinyAtariActorCritic(num_actions=num_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    return agent, optimizer


def emit_result(result: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


if __name__ == "__main__":
    main()
