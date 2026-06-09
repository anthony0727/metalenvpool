# Design Notes

MetalEnvPool is a small translation of the EnvPool idea to Apple GPU execution.

## Target

The target is high-throughput environment stepping for RL experiments on Apple Silicon:

1. keep env state in a fixed set of tensors
2. step all env instances in one batched tensor program
3. reset completed slots from a device-side done mask
4. avoid `.cpu()`, `.numpy()`, and `.item()` in the hot path
5. expose Gymnasium spaces as metadata, not as the speed-critical API
6. benchmark with explicit device synchronization

## Apple GPU Stack

Metal is the low-level compute and graphics API. Metal Performance Shaders and MPSGraph are optimized libraries on top of Metal. PyTorch's `mps` device uses those MPS/MPSGraph layers and, where supported, custom Metal kernels.

That means this repo has two backend lanes:

- Torch-free native env runtime: metal-cpp, `MTLCommandQueue`, shared `MTLBuffer`s, and `.metallib` compiled from packaged MSL.
- Torch-compatible training runtime: `torch.mps.compile_shader`, used when rollout tensors must stay directly readable by a PyTorch MPS learner.

The active fused kernels live under `src/metalenvpool/native/`. The native benchmark compiles them to `.metallib`; the Torch-compatible path compiles the same source at runtime through `torch.mps.compile_shader`.

Relevant PyTorch MPS controls:

- `torch.backends.mps.is_available()` selects the device.
- `torch.mps.synchronize()` is required for honest wall-clock timing.
- `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` expose memory counters.
- `PYTORCH_MPS_FAST_MATH=1` may speed some MPS kernels.
- `PYTORCH_MPS_PREFER_METAL=1` can prefer Metal kernels over MPSGraph for some matmul paths.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` allows CPU fallback for unsupported ops, but it can silently harm benchmark validity.

## EnvPool Translation

EnvPool uses a C++ execution engine, batched action/state buffers, and fast compatibility layers. MetalEnvPool keeps the same shape of idea but starts from a tensor-native contract:

- `reset(seed, env_ids=None)` initializes all or selected env slots where a backend supports selected reset.
- `reset_done(done_tensor)` autoresets selected env slots without a Python sync.
- `step(actions)` advances the whole batch.
- `send(actions)` and `recv()` offer an EnvPool-like split call.
- `sample_random_actions()` generates random actions on the env device.

The first version intentionally does not optimize for Gymnasium or SB3 adapters. Those adapters are useful for debugging and demos, but they convert through NumPy and lose the Apple Silicon advantage. The primary API is `TensorEnv`; future PPO/DQN implementations should target it directly.

## Memory Layout

The repo's distinctive layout is the rollout observation ABI:

```text
[T, N, *obs_shape] -> [T * N, *obs_shape]
```

The first shape is the writer layout for env kernels. The second shape is the learner view. The view is zero-copy when the rollout buffer is contiguous. For Atari, the fused Metal kernel writes directly into `[T, N, 4, 84, 84]` uint8 observations, so a DQN/PPO learner can read `[T * N, 4, 84, 84]` without Python frame lists, NumPy stacks, or transposes.

See `docs/MEMORY_LAYOUT.md` for the offset formula and tests.

## Current Limits

- PyTorch MPS op coverage is the binding constraint.
- Direct metal-cpp to Torch MPS tensor interop is not available through a stable public PyTorch Python API.
- Dynamic Python control around done masks can force synchronization, so automatic reset is not in the hot path.
- Rendering belongs outside the tensor engine.
