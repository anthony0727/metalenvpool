# MetalEnvPool

EnvPool-style batched reinforcement learning environments for Apple GPU machines.

The native environment benchmark does not depend on Torch. It uses metal-cpp, `MTLCommandQueue`, shared `MTLBuffer`s, and packaged Metal shaders. The Torch-compatible path remains available for training integration, where the rollout tensor must be directly readable by a PyTorch learner.

The Apple-specific idea is a tensor-native env contract plus a learner-ready rollout memory layout. Environment kernels step batched state tensors on one device, reset completed slots from device-side masks, and write directly into contiguous `[T, N, *obs]` buffers. PPO/DQN code consumes the same storage as `[T * N, *obs]` without Python object conversion.

## Why Metal, MPS, and EnvPool

Apple's stack is layered:

- Metal is the low-level GPU API.
- Metal Performance Shaders and MPSGraph provide optimized compute kernels and graph execution on top of Metal.
- PyTorch's `mps` device maps tensor operations to MPS/MPSGraph and custom Metal kernels.

EnvPool's key idea is that environment stepping should be a high-throughput engine instead of a Python loop. MetalEnvPool translates that idea to Apple Silicon by batching many env instances into one tensor program.

## Quick Start

```bash
uv run --extra test pytest -q
scripts/build-native-metalcpp
build/metalenvpool-native/metalenvpool_atari_bench --num-envs 16 --rollout-steps 128 --iters 1000 --warmup 50
uv run metalenvpool-build-native --output-dir build/metalenvpool-native
uv run --extra dev python examples/probe_torch_mps_interop.py
uv run --extra torch python examples/bench_atari_pipeline.py --num-envs 64 --rollout-steps 128 --iters 1000 --device auto
uv run --extra torch --extra atari python examples/bench_atari_suite.py --num-envs 16 --rollout-steps 128 --iters 1000 --devices auto --format markdown
uv run --extra torch python examples/bench_point.py --num-envs 65536 --steps 1000 --device auto --action-mode random
uv run --extra torch python examples/train_metalpoint_ppo.py --num-envs 8192 --num-steps 64 --total-timesteps 4194304 --device auto
uv run --extra video python examples/train_sb3_rollout_video.py --env-id InvertedDoublePendulum-v5 --num-envs 8 --total-timesteps 1000000 --device cpu
uv run python examples/bench_gymnasium_vector.py --env-id CartPole-v1 --num-envs 16 --steps 10000
uv run --extra bench python examples/bench_sb3_ppo.py --env-id CartPole-v1 --device cpu
```

On Apple Silicon, `--device auto` selects `mps` when PyTorch reports it as available. On other machines it uses CPU.

The Atari suite requires ALE ROMs installed in the active environment:

```bash
uv run --extra atari --with 'autorom[accept-rom-license]' AutoROM --accept-license
```

## Benchmarks

Atari preprocessing shape: `N=16`, `T=128`, raw RGB `210x160x3`, output `[T, N, 4, 84, 84]` uint8, 1,000 timed kernel launches after 50 warmup launches.

| Backend | Torch required | Sync policy | Stacked obs/s | Raw frames/s | vs CPU |
| --- | ---: | --- | ---: | ---: | ---: |
| metal-cpp + shared `MTLBuffer` | No | wait after final launch | 413,035 | 826,070 | 17.53x |
| Torch MPS `compile_shader` | Yes | `torch.mps.synchronize()` after loop | 337,356 | 674,712 | 14.32x |
| metal-cpp + shared `MTLBuffer` | No | wait every launch | 65,521 | 131,041 | 2.78x |
| Torch CPU reference | Yes | CPU loop | 23,564 | 47,128 | 1.00x |

Popular ALE frame-source table, using real Gymnasium/ALE RGB frames but timing only fused preprocessing and rollout writes:

| Env | MPS stacked obs/s | CPU stacked obs/s | MPS/CPU |
| --- | ---: | ---: | ---: |
| Breakout | 320,917 | 36,059 | 8.90x |
| Pong | 309,171 | 36,572 | 8.45x |
| SpaceInvaders | 299,618 | 37,066 | 8.08x |
| Seaquest | 309,710 | 35,732 | 8.67x |
| MsPacman | 342,594 | 37,600 | 9.11x |
| Qbert | 340,252 | 37,644 | 9.04x |
| BeamRider | 326,088 | 38,157 | 8.55x |
| Enduro | 338,728 | 38,016 | 8.91x |

These are not full RL training scores. They isolate the env preprocessing and rollout-write path.

Full PPO training checks:

| Lane | Task | Device | Timesteps | Steps/sec | Result |
| --- | --- | --- | ---: | ---: | --- |
| MetalEnvPool tensor PPO | PointMass `TensorEnv` | MPS | 4,194,304 | 834,506 | custom env, full rollout/update loop |
| Stable-Baselines3 PPO | `InvertedDoublePendulum-v5` | CPU | 1,000,000 | 18,688 | eval mean 9,323; 1,000-step MP4 rollout |

The first row is the speed lane. The second row is the public-env trained-rollout demo. They are not the same task.

Tracked demo artifacts:

- `artifacts/ppo_rollouts/InvertedDoublePendulum-v5_ppo_rollout.mp4`
- `artifacts/ppo_rollouts/InvertedDoublePendulum-v5_ppo.zip`
- `artifacts/ppo_rollouts/InvertedDoublePendulum-v5_ppo_result.json`

## Fast API

```python
from metalenvpool import MetalPointPool, PointConfig, check_tensor_env

pool = MetalPointPool(PointConfig(num_envs=65536), device="auto")
obs = pool.reset(seed=7)
actions = pool.sample_random_actions()
step = pool.step(actions)
obs = pool.reset_done(step.terminated | step.truncated)

print(step.obs.shape, step.reward.shape, step.terminated.shape)
check_tensor_env(pool)
```

The hot path is intentionally stricter than Gymnasium:

- observation: `[num_envs, obs_dim]`
- action: `[num_envs, action_dim]`
- reward: `[num_envs]`
- terminated/truncated: `[num_envs]`
- all fields are tensors on the env device
- autoreset uses `reset_done(done_tensor)` to avoid CPU syncs
- PPO code should not call `.cpu()`, `.numpy()`, or `.item()` inside rollouts

Gymnasium spaces are exposed as metadata for integration, but the primary API is `TensorEnv`. This is the extension point for custom Apple Silicon envs: implement `reset`, `reset_done`, `step`, `sample_random_actions`, and `zero_actions` over batched tensors.

For Torch rollout-based learners, `MetalRolloutBuffer.learner_obs_view()` exposes observations as `[T * N, *obs_shape]` without copying inside Torch. The exact ABI is documented in `docs/MEMORY_LAYOUT.md`.

## Unique Memory Layout

The core layout is:

```text
writer:  [T, N, *obs_shape]
learner: [T * N, *obs_shape]
```

For Atari this is `[T, N, 4, 84, 84]` uint8. The Metal preprocessing kernel writes directly into that rollout buffer after max-pool, grayscale, resize, and frame-stack update. That is the Apple Silicon angle: use unified memory and fixed strides so CPU-side frame producers, Metal kernels, and the learner agree on one buffer contract.

## Current Backends

Torch-compatible Python backends:

- `MetalPointPool`: one packaged Metal kernel per step on `mps`
- state `[x, y, vx, vy]`, target `[tx, ty]`, action `[ax, ay]`
- useful for measuring the ceiling of env stepping when the transition kernel is narrow enough to fuse

- `MetalAtariPreprocessor`:
- accepts raw Atari RGB frames
- fuses max-pool, grayscale, resize, frame-stack, and rollout-buffer write
- writes directly into `[T, N, 4, 84, 84]` uint8 observations for the learner
- benchmarks the same path across common ALE games with `examples/bench_atari_suite.py`

Torch-free native backend:

- `native/metalenvpool_atari_bench.cpp`
- built by `scripts/build-native-metalcpp`
- uses Apple's metal-cpp headers, `MTLDevice`, `MTLCommandQueue`, `MTLComputePipelineState`, and shared `MTLBuffer`s
- proves the env/preprocess runtime can run without importing Torch

Torch interop probe:

- `examples/probe_torch_mps_interop.py`
- builds a C++ extension against PyTorch's MPS headers
- verifies Torch MPS rollout tensors are shared-storage backed
- confirms the stable allocator interface exposes a CPU mapping, not the `id<MTLBuffer>` object needed for direct metal-cpp dispatch over Torch-owned tensors

Metal shader sources live under `src/metalenvpool/native/`:

- `atari_preprocess.metal`
- `point_step.metal`

The Torch-compatible backend compiles those sources through `torch.mps.compile_shader` so the learner can consume Torch MPS tensors without a copy. The native metal-cpp backend compiles the same sources to `.metallib` and owns Metal buffers directly. See `docs/NATIVE_BACKEND.md`.

To compile the packaged shaders with Apple's Metal Toolchain:

```bash
uv run metalenvpool-build-native --output-dir build/metalenvpool-native
```

```bash
uv run --extra torch python examples/bench_atari_pipeline.py --num-envs 64 --rollout-steps 128 --iters 1000 --device auto
uv run --extra torch --extra atari python examples/bench_atari_suite.py --num-envs 16 --rollout-steps 128 --iters 1000 --devices auto --format markdown
uv run --extra torch python examples/bench_point.py --num-envs 65536 --steps 10000 --device auto
uv run --extra torch python examples/train_metalpoint_ppo.py --num-envs 8192 --num-steps 64 --total-timesteps 1048576 --device auto
```

On the local Apple Silicon verification run, the fused PointMass shader reached about 220M env-steps/sec with fixed actions and 174M env-steps/sec with random actions. See `docs/BENCHMARKS.md`.

The real benchmark target is full RL training throughput against Gymnasium, Stable-Baselines3, CleanRL, LeanRL-style baselines, and EnvPool where the setup is comparable. See `docs/COMPETITIVE_BENCHMARKS.md`.

## References

- EnvPool: https://github.com/sail-sg/envpool
- PyTorch MPS backend: https://docs.pytorch.org/docs/stable/notes/mps.html
- `torch.mps` API: https://docs.pytorch.org/docs/main/mps.html
- Apple PyTorch on Metal: https://developer.apple.com/metal/pytorch/
