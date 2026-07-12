# Benchmarks

Benchmarks use `torch.mps.synchronize()` at timing boundaries when running on MPS.

## Local Apple Silicon Run

PyTorch reported:

- torch: 2.12.0
- MPS built: true
- MPS available: true
- `torch.mps.compile_shader`: true

## Atari Preprocessing To Rollout Buffer

Native Torch-free metal-cpp comparison:

```bash
scripts/build-native-metalcpp
build/metalenvpool-native/metalenvpool_atari_bench --num-envs 16 --rollout-steps 128 --iters 1000 --warmup 50
uv run --extra torch python examples/bench_atari_pipeline.py --num-envs 16 --rollout-steps 128 --iters 1000 --warmup 50 --device mps --source synthetic
uv run --extra torch python examples/bench_atari_pipeline.py --num-envs 16 --rollout-steps 128 --iters 1000 --warmup 50 --device cpu --source synthetic
```

| Backend | Torch required | Sync policy | Stacked obs/s | Raw frames/s | vs CPU |
| --- | ---: | --- | ---: | ---: | ---: |
| metal-cpp + shared `MTLBuffer` | No | wait after final launch | 413,035 | 826,070 | 17.53x |
| Torch MPS `compile_shader` | Yes | `torch.mps.synchronize()` after loop | 337,356 | 674,712 | 14.32x |
| metal-cpp + shared `MTLBuffer` | No | wait every launch | 65,521 | 131,041 | 2.78x |
| Torch CPU reference | Yes | CPU loop | 23,564 | 47,128 | 1.00x |

This table uses synthetic Atari-shaped RGB frames and the same `[128, 16, 4, 84, 84]` uint8 rollout layout. It shows why the native runtime must queue command buffers and avoid per-step synchronization.

Command:

```bash
uv run --extra torch python examples/bench_atari_pipeline.py --num-envs 64 --rollout-steps 128 --iters 1000 --warmup 50 --device mps --source synthetic
```

Result:

- device: `mps`
- backend: custom Metal shader via PyTorch runtime shader compiler
- fused stages: max-pool, grayscale, resize to 84x84, 4-frame stack, rollout write
- rollout layout: `[128, 64, 4, 84, 84]` uint8
- seconds: 0.164
- raw frames/sec: 781,066
- stacked observations/sec: 390,533

CPU reference comparison:

```bash
uv run --extra torch python examples/bench_atari_pipeline.py --num-envs 64 --rollout-steps 128 --iters 1000 --warmup 50 --device cpu --source synthetic
```

- raw frames/sec: 83,385
- stacked observations/sec: 41,692

Actual `ALE/Breakout-v5` frame source after installing ROMs locally:

```bash
uv run --extra torch --extra atari python examples/bench_atari_pipeline.py --num-envs 16 --rollout-steps 128 --iters 1000 --warmup 50 --device mps --source gym --env-id ALE/Breakout-v5
```

- MPS stacked observations/sec: 315,761
- MPS raw frames/sec: 631,521
- CPU stacked observations/sec: 28,982
- CPU raw frames/sec: 57,964
- speedup: about 10.9x on the preprocessing/write path

The timed section reuses real Breakout frames captured through Gymnasium/ALE; it does not yet include continuous emulator stepping in the timed loop.

Popular ALE suite:

```bash
uv run --extra atari --with 'autorom[accept-rom-license]' AutoROM --accept-license
uv run --extra torch --extra atari python examples/bench_atari_suite.py --num-envs 16 --rollout-steps 128 --iters 1000 --warmup 50 --devices mps,cpu --format markdown
```

| Env | Action space | MPS stacked obs/s | CPU stacked obs/s | MPS/CPU | MPS raw frames/s | CPU raw frames/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALE/Breakout-v5 | Discrete(4) | 307,823 | 35,757 | 8.61x | 615,646 | 71,515 |
| ALE/Pong-v5 | Discrete(6) | 306,079 | 35,807 | 8.55x | 612,158 | 71,614 |
| ALE/SpaceInvaders-v5 | Discrete(6) | 353,620 | 34,945 | 10.12x | 707,241 | 69,891 |
| ALE/Seaquest-v5 | Discrete(18) | 325,654 | 35,294 | 9.23x | 651,309 | 70,588 |
| ALE/MsPacman-v5 | Discrete(9) | 358,981 | 35,639 | 10.07x | 717,961 | 71,278 |
| ALE/Qbert-v5 | Discrete(6) | 318,711 | 35,883 | 8.88x | 637,422 | 71,766 |
| ALE/BeamRider-v5 | Discrete(9) | 332,890 | 36,069 | 9.23x | 665,781 | 72,138 |
| ALE/Enduro-v5 | Discrete(9) | 316,369 | 35,500 | 8.91x | 632,738 | 71,000 |

This table measures the fused Atari preprocessing and rollout-write path using real Gymnasium/ALE RGB frames. It does not include emulator stepping, policy inference, backpropagation, or optimizer work.

Native `xcrun metal` is installed on this machine and the packaged MSL sources compile to `.metallib`:

```bash
uv run metalenvpool-build-native --output-dir build/metalenvpool-native
```

The active runtime still uses `torch.mps.compile_shader` because it can operate directly on PyTorch MPS tensors. A native metal-cpp/pybind11 runtime would need an equally clean tensor/buffer interop story before it replaces that path.

### Fused PointMass

The benchmark auto-resets terminated and truncated slots on every step. Older
results produced before this invariant was enforced timed inactive slots after
the warm-up reached `PointConfig.max_steps`; those numbers are intentionally no
longer treated as valid environment-throughput evidence.

Command:

```bash
uv run python examples/bench_point.py --num-envs 65536 --steps 1000 --warmup 200 --device mps --action-mode zero --seed 7
```

Result:

- aggregation: median of 3 consecutive runs (range shown below)
- device: `mps`
- backend: custom Metal shader
- env steps: 65,536,000
- median seconds: 0.536
- median throughput: 122,310,140 env-steps/sec
- throughput range: 89,287,193–127,481,732 env-steps/sec
- current allocated memory: 2,752,512 bytes

CPU comparison:

```bash
uv run python examples/bench_point.py --num-envs 65536 --steps 1000 --warmup 200 --device cpu --action-mode zero --seed 7
```

- median throughput: 24,424,726 env-steps/sec
- throughput range: 23,989,920–24,589,355 env-steps/sec

Random-action MPS run:

```bash
uv run python examples/bench_point.py --num-envs 65536 --steps 1000 --warmup 200 --device mps --action-mode random --seed 7
```

- median throughput: 92,262,469 env-steps/sec
- throughput range: 73,284,754–106,173,930 env-steps/sec

CPU random-action comparison:

- median throughput: 23,353,427 env-steps/sec
- throughput range: 22,514,388–23,729,398 env-steps/sec

Median MPS/CPU speedup is 5.01x for zero actions and 3.95x for random
actions. These are comparisons of the same active PointMass transition and
autoreset path on this machine, not comparisons with EnvPool or a public RL
task.

The absolute number varies by run and dependency environment. These rows include
device-side autoreset work and keep all timed slots active; the fused Metal path
remains materially faster than the torch CPU path for this simple transition.

## EnvPool Paper Task Matrix

EnvPool's isolated execution benchmark centers on three exact task names:
Atari Pong-v5, MuJoCo Ant-v3, and dm_control cheetah run. This repo now runs a
separate matrix for those exact names rather than substituting arbitrary popular
Gymnasium tasks.

Command:

```bash
uv run --extra envpool-paper python examples/run_envpool_paper_tasks.py \
  --out-dir artifacts/envpool_paper_tasks/2026-06-09 \
  --physics-steps 200000 \
  --pong-iters 1000
```

Results:

| Paper task | Local id | Status | MetalEnvPool coverage | Throughput |
| --- | --- | --- | --- | ---: |
| Atari Pong-v5 | `ALE/Pong-v5` | ok | Atari RGB preprocessing and rollout write only | MPS 352,862 stacked obs/s; CPU 36,709 |
| MuJoCo Ant-v3 | `Ant-v3` | blocked | not implemented | - |
| dm_control cheetah run | `suite.load("cheetah", "run")` | ok | not implemented; CPU reference only | 52,862 steps/s |

Tracked outputs:

- `artifacts/envpool_paper_tasks/2026-06-09/summary.json`
- `artifacts/envpool_paper_tasks/2026-06-09/summary.md`
- `artifacts/envpool_paper_tasks/2026-06-09/pong.json`
- `artifacts/envpool_paper_tasks/2026-06-09/ant.json`
- `artifacts/envpool_paper_tasks/2026-06-09/cheetah_run.json`

Interpretation: MetalEnvPool does not yet reproduce EnvPool's paper benchmark.
It covers Pong preprocessing/write for exact Pong frames, but not ALE emulator
execution, Ant-v3 dynamics, or dm_control cheetah dynamics.

### Local EnvPool Reference

Command:

```bash
uv run --extra envpool-compare python examples/run_envpool_reference.py \
  --out-dir artifacts/envpool_reference/2026-06-09 \
  --num-envs 1 16 64 \
  --steps-small 5000 \
  --steps-large 2000
```

Results:

| Task | EnvPool 1 env | EnvPool 16 envs | EnvPool 64 envs | MetalEnvPool row |
| --- | ---: | ---: | ---: | --- |
| `Pong-v5` | 3,750 steps/s | 18,347 steps/s | 22,128 steps/s | 352,862 stacked obs/s, preprocessing only |
| `Ant-v3` | 6,985 steps/s | 37,005 steps/s | 47,039 steps/s | no full env row |
| `CheetahRun-v1` | 36,405 steps/s | 252,518 steps/s | 517,870 steps/s | no full env row |

Win assessment: `not_won`. The Pong Metal path is faster than EnvPool's full
Pong env execution number but it is not the same work; it excludes ALE emulator
execution.

## Full PPO Training Checks

### Exact `InvertedDoublePendulum-v5` SB3/Gymnasium Comparison

These rows use the same public Gymnasium/MuJoCo task and the same SB3 PPO
script. This is the fair public-task comparison.

Commands:

```bash
uv run --extra video python examples/colab_sb3_gymnasium_benchmark.py \
  --env-id InvertedDoublePendulum-v5 \
  --num-envs 8 \
  --total-timesteps 50000 \
  --device cpu \
  --vec-env dummy \
  --n-steps 512 \
  --batch-size 1024 \
  --n-epochs 10 \
  --no-install \
  --out runs/mac_idp_cpu_50k.json

uv run --extra video python examples/colab_sb3_gymnasium_benchmark.py \
  --env-id InvertedDoublePendulum-v5 \
  --num-envs 8 \
  --total-timesteps 50000 \
  --device mps \
  --vec-env dummy \
  --n-steps 512 \
  --batch-size 1024 \
  --n-epochs 10 \
  --no-install \
  --out runs/mac_idp_mps_50k.json
```

Results:

| Runner | Runtime | Model device | Timesteps | Throughput | Notes |
| --- | --- | --- | ---: | ---: | --- |
| SB3 PPO | Apple M4 SoC CPU backend | CPU | 50,000 | 17,223 steps/sec | exact Gymnasium/MuJoCo task |
| SB3 PPO | Apple M4 SoC MPS GPU backend | MPS | 50,000 | 3,077 steps/sec | exact task, SB3 warned MLP PPO underutilizes GPU |
| SB3 PPO | Colab TPU v5e1 | CPU | 50,000 | 1,517 steps/sec | TPU runtime exposed `torch_xla`/`xla:0`, but SB3 resolved to CPU |

Tracked outputs:

- `artifacts/rl_hardware_matrix/2026-06-09/idp_mac-cpu.json`
- `artifacts/rl_hardware_matrix/2026-06-09/idp_mac-mps.json`
- `artifacts/rl_hardware_matrix/2026-06-09/idp_tpu-v5e1.json`
- `artifacts/rl_hardware_matrix/2026-06-09/hardware_matrix.json`

Interpretation: on the exact public task, this repo does not yet show a
MetalEnvPool win. SB3/Gymnasium is memory/sync heavy and MLP PPO underutilizes
GPU backends; MPS loses to the Apple M4 SoC CPU backend here. A fair MetalEnvPool
win requires implementing the same task dynamics as a tensor-native env, not
comparing against a custom PointMass task.

### Representative RL Hardware Matrix

The broader public-task control matrix covers `CartPole-v1`, `Pendulum-v1`, and
`InvertedDoublePendulum-v5` with the same SB3 PPO runner across the Apple M4 SoC
CPU backend, Apple M4 SoC MPS GPU backend, Colab CUDA GPU, and Colab TPU-runtime
CPU fallback:

- results: `artifacts/rl_hardware_matrix/2026-06-09/hardware_matrix.md`
- doc: `docs/RL_HARDWARE_MATRIX.md`

The matrix is a hardware/framework control. It is not a tensor-native
MetalEnvPool win claim.

### Custom Tensor-Native PPO On MetalPointPool

Command:

```bash
uv run --extra torch python examples/train_metalpoint_ppo.py \
  --num-envs 8192 \
  --num-steps 64 \
  --total-timesteps 4194304 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --device auto
```

Result:

- backend: `metalenvpool-point-ppo`
- device: `mps`
- fused env shader: true
- total timesteps: 4,194,304
- aggregation: median of 4 runs in the current dependency environment
- median seconds: 6.022
- median throughput: 696,638 env-steps/sec
- throughput range: 501,216–753,100 env-steps/sec
- mean rollout reward at final iteration: -3.951

This is a full PPO rollout/update loop over the tensor-native PointMass task,
not a fair `InvertedDoublePendulum-v5` comparison. It is the speed demonstration
for the `TensorEnv` API: actions, observations, rewards, done masks, autoreset,
rollout tensors, and learner updates stay on MPS during the hot path. The fixed
seed produced the same final rollout reward in all four runs; this short run is
throughput evidence, not evidence that PPO learned a strong policy.

### Custom Tensor-Native PPO On MetalMPESimplePool

Command:

```bash
uv run --extra torch python examples/train_mpe_simple_ppo.py \
  --num-envs 8192 \
  --num-steps 64 \
  --total-timesteps 4194304 \
  --num-minibatches 8 \
  --update-epochs 2 \
  --device auto
```

Result:

- backend: `metalenvpool-mpe-simple-ppo`
- device: `mps`
- fused env shader: true
- total timesteps: 4,194,304
- seconds: 3.981
- throughput: 1,053,625 env-steps/sec
- first mean rollout reward: -1.050
- final mean rollout reward: -0.841
- deterministic eval mean episode return after training: -19.306

`MetalMPESimplePool` is a tensor-native MPE Simple-style particle task, not a
full PettingZoo wrapper. It uses one agent, one landmark, observation
`[vx, vy, landmark_rel_x, landmark_rel_y]`, and continuous action channels
`[noop, left, right, down, up]`.

### Local PPO Baselines

These are executable local references, not exact task-matched comparisons.

| Runner | Task | Device | Timesteps | Throughput |
| --- | --- | --- | ---: | ---: |
| LeanRL `ppo_continuous_action.py` | `Pendulum-v1` | CPU | 131,072 | 13,850 steps/sec |
| LeanRL `ppo_continuous_action_torchcompile.py --compile` | `Pendulum-v1` | CPU | 131,072 | 5,100 steps/sec |
| Stable-Baselines3 PPO | `Pendulum-v1` | CPU | 1,048,576 | 34,555 steps/sec |

LeanRL's published headline numbers are CUDA/H100-oriented. On this Mac, the
local LeanRL `torchcompile` CPU path is slower than the plain script because it
does not get the CUDA graph path that LeanRL is designed around.

### Colab SB3/Gymnasium Commands

Command:

```bash
colab run --gpu T4 examples/colab_sb3_gymnasium_benchmark.py \
  --env-id InvertedDoublePendulum-v5 \
  --num-envs 8 \
  --total-timesteps 50000 \
  --device auto \
  --vec-env dummy \
  --n-steps 512 \
  --batch-size 1024 \
  --n-epochs 10

colab run --tpu v5e1 examples/colab_sb3_gymnasium_benchmark.py \
  --env-id InvertedDoublePendulum-v5 \
  --num-envs 8 \
  --total-timesteps 50000 \
  --device auto \
  --vec-env dummy \
  --n-steps 512 \
  --batch-size 1024 \
  --n-epochs 10
```

The measured Colab outputs are included in the exact public-task table above.

### Trained Public-Env Rollout

Command:

```bash
uv run --extra video python examples/train_sb3_rollout_video.py \
  --env-id InvertedDoublePendulum-v5 \
  --num-envs 8 \
  --total-timesteps 1000000 \
  --device cpu \
  --vec-env dummy \
  --n-steps 512 \
  --batch-size 1024 \
  --n-epochs 10 \
  --eval-episodes 3 \
  --max-rollout-steps 1000 \
  --out-dir runs/ppo_rollouts
```

Result:

- backend: Stable-Baselines3 PPO
- task: `InvertedDoublePendulum-v5`
- total timesteps: 1,000,000
- seconds: 53.511
- throughput: 18,688 env-steps/sec
- eval returns: 9323.43, 9323.21, 9323.43
- eval mean return: 9323.36
- recorded rollout: 1,000 steps, return 9323.99
- video: `runs/ppo_rollouts/InvertedDoublePendulum-v5_ppo_rollout.mp4`
- tracked demo copy: `artifacts/ppo_rollouts/InvertedDoublePendulum-v5_ppo_rollout.mp4`
- tracked model copy: `artifacts/ppo_rollouts/InvertedDoublePendulum-v5_ppo.zip`
- tracked result JSON: `artifacts/ppo_rollouts/InvertedDoublePendulum-v5_ppo_result.json`

## Interpretation

The result is now two separate claims:

- A fused Metal kernel is the correct Apple-GPU path for narrow env transitions.
- Atari preprocessing is a credible Apple Silicon target because frame max-pooling, grayscale, resize, stacking, and rollout writes are memory-layout heavy.
- Tensor-native PPO on `MetalPointPool` is much faster than local Python-framework PPO baselines, but it is not a task-matched public benchmark.
- SB3 PPO solves `InvertedDoublePendulum-v5` and provides the public-env rollout video, but it is not the MetalEnvPool speed path.
- Current M-series memory capacity is a practical limit for large deep RL training jobs. The point of this repo is to prepare the runtime and rollout-buffer path for Apple training hardware with larger usable memory.
- The next high-value step is a public task whose dynamics can run as a tensor-native `TensorEnv`, so the speed comparison and public-environment comparison become the same row.
