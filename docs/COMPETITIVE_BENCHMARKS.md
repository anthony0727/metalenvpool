# Competitive Benchmarks

MetalEnvPool should be judged against full RL training throughput, not isolated shader speed.

## Win Definition

A real win requires a matched setup:

- same environment or statistically equivalent environment family
- same algorithm and rollout/update accounting
- same observation/action preprocessing
- same total timesteps and evaluation schedule
- same machine, same thermal state window, same seed set
- wall-clock measured from first reset to final policy update
- reward curve reported, not only samples/sec

The primary metric is:

```text
time_to_reward_threshold
```

Secondary metrics:

- environment steps/sec
- policy updates/sec
- total train seconds
- final/eval return
- memory footprint
- CPU utilization and MPS/Metal memory counters

## Baseline Lanes

### Lane 1: Gymnasium Vector Stepping

Purpose: measure conventional Python/vector env overhead.

```bash
uv run python examples/bench_gymnasium_vector.py --env-id CartPole-v1 --num-envs 16 --steps 10000 --mode sync
uv run python examples/bench_gymnasium_vector.py --env-id CartPole-v1 --num-envs 16 --steps 10000 --mode async
```

### Lane 2: Stable-Baselines3 PPO

Purpose: compare against a familiar PyTorch PPO training stack.

```bash
uv run --extra bench python examples/bench_sb3_ppo.py \
  --env-id CartPole-v1 \
  --num-envs 8 \
  --total-timesteps 25000 \
  --vec-env dummy \
  --device cpu
```

On Apple Silicon, test both CPU and MPS learner devices when supported by the library:

```bash
uv run --extra bench python examples/bench_sb3_ppo.py --env-id CartPole-v1 --device cpu
uv run --extra bench python examples/bench_sb3_ppo.py --env-id CartPole-v1 --device mps
```

Local sanity result on CartPole-v1, 8 envs, 25k timesteps:

- SB3 PPO + DummyVecEnv + CPU: 30,724 env-steps/sec
- SB3 PPO + SubprocVecEnv + CPU: 23,108 env-steps/sec
- SB3 PPO + DummyVecEnv + MPS: 2,088 env-steps/sec

SB3 warns that PPO with an MLP policy is primarily intended for CPU and may be slower on GPU. This means CartPole MLP PPO is a CPU-control baseline, not a good MPS-learner showcase.

### Lane 3: CleanRL / LeanRL

Purpose: compare against single-file optimized PPO implementations.

This repo should not vendor those projects casually. The benchmark runner should call checked-out upstream scripts with pinned commit SHAs and capture JSON/CSV output. Comparable setup means the same env, same network, same rollout length, same minibatch size, same update count, and same seed set.

Local check:

| Runner | Task | Device | Timesteps | Throughput |
| --- | --- | --- | ---: | ---: |
| MetalEnvPool tensor PPO | PointMass `TensorEnv` | MPS | 4,194,304 | 834,506 steps/sec |
| LeanRL PPO | `Pendulum-v1` | CPU | 131,072 | 13,850 steps/sec |
| LeanRL PPO + `torch.compile` | `Pendulum-v1` | CPU | 131,072 | 5,100 steps/sec |
| Stable-Baselines3 PPO | `Pendulum-v1` | CPU | 1,048,576 | 34,555 steps/sec |

This wins the local LeanRL-style training-speed check, but it is not a
task-matched public benchmark because the fast lane uses a custom tensor-native
PointMass task. Treat it as evidence that the API shape works, not as a public
MuJoCo/Atari leaderboard result.

### Lane 4: Atari / Breakout

Purpose: test image-heavy policy learning where MPS may matter.

Breakout is a later lane because Atari requires ALE/ROM setup and exact preprocessing:

- `AtariPreprocessing`
- frame skip
- grayscale/resize
- frame stacking
- no-op reset convention
- episodic-life convention where applicable

Do not claim a Breakout win until preprocessing and scoring match the baseline implementation exactly.

## Apple SoC Hypothesis

The plausible advantage is unified memory and local bandwidth:

- env buffers, rollout buffers, and learner tensors can avoid PCIe-style transfer costs
- CPU env stepping and MPS learner updates can share memory pressure more efficiently than discrete CPU/GPU systems
- fused Metal kernels can remove Python and eager-MPS launch overhead for simple transition dynamics

The plausible failure mode is equally clear:

- many env transitions are branchy and tiny
- eager MPS can be slower than CPU for small fragmented kernels
- training can silently synchronize if code calls `.cpu()`, `.numpy()`, or `.item()` in the hot path

## Low-Level Metal Roadmap

The Python/PyTorch MPS layer is the compatibility layer. The competitive path likely needs low-level Metal for hot env kernels:

1. fused reset/step kernels for simple continuous-control envs
2. fused rollout-buffer write kernels
3. batched reward/done/autoreset kernels
4. optional observation preprocessing kernels for image envs
5. policy inference/training stays in PyTorch MPS until profiling proves otherwise

Do not rewrite PPO in Metal first. The first low-level target is env stepping plus buffer writes, because that is where Python vector envs waste wall-clock on Mac.

## Next Required MetalEnvPool Lane

The next implementation should be an end-to-end PPO/DQN runner for a single tensor-native env lane:

- a public comparable task where the env dynamics can stay as tensors
- rollout buffer tensors in learner-ready layout
- PyTorch MPS policy/value network
- no `.cpu()`, `.numpy()`, or `.item()` in the rollout/update hot path
- final JSON with train seconds, updates/sec, env-steps/sec, and reward curve

Only after that exists can we compare fairly against SB3/CleanRL/LeanRL/EnvPool on a matched public task.

Gymnasium/SB3 compatibility is not the speed target. Their adapters are useful
for sanity checks and rendered demos, but the benchmark lane must target
`TensorEnv` directly so actions, rewards, done masks, autoreset, and rollout
storage stay on the Apple device.
