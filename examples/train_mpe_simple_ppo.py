"""PPO training benchmark for the fused MPE Simple-style Metal env."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.distributions.normal import Normal

from metalenvpool import MetalMPESimplePool, MPESimpleConfig, resolve_device, synchronize


def layer_init(layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size), std=2**0.5),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size), std=2**0.5),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size), std=2**0.5),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size), std=2**0.5),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        raw_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if raw_action is None:
            raw_action = mean + std * torch.randn_like(mean)
        return raw_action, dist.log_prob(raw_action).sum(1), dist.entropy().sum(1), self.critic(obs)

    def deterministic_env_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.actor_mean(obs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=1_048_576)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--world-size", type=float, default=1.0)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("num-envs and num-steps must be positive")
    batch_size = args.num_envs * args.num_steps
    if args.num_minibatches <= 0 or batch_size % args.num_minibatches != 0:
        raise ValueError("num-minibatches must divide num-envs * num-steps")
    num_iterations = max(1, args.total_timesteps // batch_size)
    measured_timesteps = num_iterations * batch_size

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    cfg = MPESimpleConfig(num_envs=args.num_envs, world_size=args.world_size, max_cycles=args.max_cycles)
    pool = MetalMPESimplePool(cfg, device=device, use_shader=True)
    next_obs = pool.reset(seed=args.seed)
    next_done = torch.zeros((args.num_envs,), device=device, dtype=torch.bool)

    obs_dim = next_obs.shape[1]
    action_dim = pool.action_shape[1]
    agent = Agent(obs_dim, action_dim, args.hidden_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.empty((args.num_steps, args.num_envs, obs_dim), device=device)
    raw_actions = torch.empty((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.empty((args.num_steps, args.num_envs), device=device)
    rewards = torch.empty((args.num_steps, args.num_envs), device=device)
    dones = torch.empty((args.num_steps, args.num_envs), device=device, dtype=torch.bool)
    values = torch.empty((args.num_steps, args.num_envs), device=device)

    minibatch_size = batch_size // args.num_minibatches
    start = time.perf_counter()
    last_loss = torch.zeros((), device=device)
    last_grad_norm = torch.zeros((), device=device)
    first_rollout_reward = None

    for iteration in range(num_iterations):
        for step in range(args.num_steps):
            obs[step].copy_(next_obs)
            dones[step].copy_(next_done)
            with torch.no_grad():
                raw_action, logprob, _entropy, value = agent.get_action_and_value(next_obs)
            raw_actions[step].copy_(raw_action)
            logprobs[step].copy_(logprob)
            values[step].copy_(value.flatten())

            result = pool.step(torch.sigmoid(raw_action))
            rewards[step].copy_(result.reward)
            next_done = result.terminated | result.truncated
            next_obs = pool.reset_done(next_done)

        if first_rollout_reward is None:
            first_rollout_reward = rewards.mean().detach().clone()

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros((args.num_envs,), device=device)
            for t in range(args.num_steps - 1, -1, -1):
                if t == args.num_steps - 1:
                    next_nonterminal = (~next_done).float()
                    next_values = next_value
                else:
                    next_nonterminal = (~dones[t + 1]).float()
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        b_obs = obs.reshape(batch_size, obs_dim)
        b_raw_actions = raw_actions.reshape(batch_size, action_dim)
        b_logprobs = logprobs.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = values.reshape(batch_size)

        for _epoch in range(args.update_epochs):
            b_inds = torch.randperm(batch_size, device=device)
            for start_idx in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start_idx : start_idx + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_raw_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                last_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                last_loss = loss.detach()

        if iteration == num_iterations - 1:
            final_rollout_reward = rewards.mean().detach().clone()

    eval_return = evaluate(agent, cfg, device=device, seed=args.seed + 1000)
    synchronize(device)
    seconds = time.perf_counter() - start
    out = {
        "backend": "metalenvpool-mpe-simple-ppo",
        "device": str(device),
        "using_shader": pool.using_shader,
        "mpe_simple_config": asdict(pool.cfg),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "num_iterations": num_iterations,
        "total_timesteps": measured_timesteps,
        "hidden_size": args.hidden_size,
        "num_minibatches": args.num_minibatches,
        "update_epochs": args.update_epochs,
        "seconds": seconds,
        "env_steps_per_second": measured_timesteps / seconds,
        "first_mean_rollout_reward": float(first_rollout_reward.cpu()),
        "final_mean_rollout_reward": float(final_rollout_reward.cpu()),
        "eval_mean_episode_return": float(eval_return.cpu()),
        "last_loss": float(last_loss.cpu()),
        "last_grad_norm": float(last_grad_norm.cpu()),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


def evaluate(agent: Agent, cfg: MPESimpleConfig, *, device: torch.device, seed: int) -> torch.Tensor:
    eval_pool = MetalMPESimplePool(cfg, device=device, use_shader=True)
    obs = eval_pool.reset(seed=seed)
    episode_return = torch.zeros((cfg.num_envs,), device=device)
    with torch.no_grad():
        for _ in range(cfg.max_cycles):
            action = agent.deterministic_env_action(obs)
            result = eval_pool.step(action)
            episode_return += result.reward
            obs = result.obs
    return episode_return.mean()


if __name__ == "__main__":
    main()
