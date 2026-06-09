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

Command:

```bash
uv run python examples/bench_point.py --num-envs 65536 --steps 10000 --warmup 200 --device mps --action-mode zero
```

Result:

- device: `mps`
- backend: custom Metal shader
- env steps: 655,360,000
- seconds: 2.976
- throughput: 220,205,586 env-steps/sec
- current allocated memory: 2,752,512 bytes

CPU comparison:

```bash
uv run python examples/bench_point.py --num-envs 65536 --steps 10000 --warmup 200 --device cpu --action-mode zero
```

- throughput: 60,816,787 env-steps/sec

Random-action MPS run:

```bash
uv run python examples/bench_point.py --num-envs 65536 --steps 1000 --warmup 100 --device mps --action-mode random
```

- throughput: 174,068,617 env-steps/sec

CPU random-action comparison:

- throughput: 50,977,453 env-steps/sec

After adding the SB3 benchmark dependencies, the same random-action point benchmark was rerun:

- MPS fused shader: 131,346,196 env-steps/sec
- CPU torch fallback: 27,582,702 env-steps/sec

The absolute number varies by run and dependency environment, but the direction is stable: fused Metal is much faster than torch CPU for this simple transition kernel.

## Full PPO Training Checks

### Tensor-Native PPO On MetalPointPool

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
- seconds: 5.026
- throughput: 834,506 env-steps/sec
- mean rollout reward at final iteration: -3.951

This is a full PPO rollout/update loop over the tensor-native PointMass task.
It is the speed demonstration for the `TensorEnv` API: actions, observations,
rewards, done masks, autoreset, rollout tensors, and learner updates stay on
MPS during the hot path.

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
