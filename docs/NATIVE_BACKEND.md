# Native Metal Backend

The current production path has three layers:

1. Python API and CPU fallback for portability, tests, and benchmark scripts.
2. Packaged Metal shader sources compiled through `torch.mps.compile_shader`
   on Apple Silicon.
3. A Torch-free metal-cpp benchmark binary that emits AIR/metallib artifacts,
   owns `MTLBuffer`s directly, and runs the same fused Atari preprocessing
   kernel without importing Torch.

The native env runtime uses metal-cpp for explicit command buffers, storage
modes, and buffer ownership. The Python/Torch path remains necessary when the
learner needs direct Torch MPS tensor views.

## Native Compiler Check

Compile packaged shaders with:

```bash
uv run metalenvpool-build-native --output-dir build/metalenvpool-native
```

This validates that the low-level MSL sources are accepted by Apple's Metal
Toolchain and can be packaged as `.metallib` artifacts.

Build and run the Torch-free native benchmark:

```bash
scripts/build-native-metalcpp
build/metalenvpool-native/metalenvpool_atari_bench --num-envs 16 --rollout-steps 128 --iters 1000 --warmup 50
```

The build script fetches Apple's official metal-cpp headers into ignored
`third_party/metal-cpp/` if they are not already present.

## Torch Training Interop

The current runtime uses `torch.mps.compile_shader` because it runs MSL kernels
directly over PyTorch MPS tensors. That preserves the learner-facing tensor
layout without a copy.

PyTorch is open source, and the installed headers do expose useful MPS internals:

- `ATen/mps/MPSAllocatorInterface.h`
- `ATen/mps/MPSStream.h`
- `ATen/mps/MPSDevice.h`

The repo includes `examples/probe_torch_mps_interop.py`, which builds a small
C++ extension and verifies that a rollout-shaped Torch MPS tensor is backed by
shared storage:

```bash
uv run --extra dev python examples/probe_torch_mps_interop.py
```

On this machine it reports `is_shared_buffer: true` and
`shared_storage_supported: true`.

However, `getSharedBufferPtr(ptr)` returns a CPU mapping of the shared-storage
buffer plus metadata. It does not expose the Objective-C `id<MTLBuffer>` object
needed by metal-cpp command encoders. That object exists inside PyTorch's
internal `BufferBlock`, but it is not exposed by the stable allocator interface.

A metal-cpp training backend is only a real upgrade if it can preserve this
contract across both sides:

- raw frame buffer
- current stacked observation buffer
- rollout observation buffer
- learner-readable tensor view

Until an `id<MTLBuffer>` bridge exists, the repo keeps both lanes:

- metal-cpp for Torch-free env/preprocess runtime benchmarking
- Torch MPS shader bridge for zero-copy Torch trajectory tensors

The serious next implementation is not a blind cast. It is either:

- a PyTorch C++ extension that uses an internal/unstable MPS allocator hook
  knowingly, with version guards, or
- an upstream PyTorch hook that exposes the underlying `MTLBuffer` and byte
  offset for custom Metal command encoders.

## Native Backend Target

The C++ backend preserves the same `RolloutObsLayout` ABI:

```text
Atari frames -> Metal kernel -> [T, N, C, H, W] rollout buffer
```

Current C++ objects:

- `MTLDevice`
- `MTLCommandQueue`
- `MTLLibrary`
- `MTLComputePipelineState`
- shared `MTLBuffer` handles for raw frames, current observations, and rollout
  observations

The current benchmark uses fixed Atari dimensions: `210x160 -> 84x84`,
`frame_stack=4`, and uint8 storage.

## Storage Mode Policy

Apple Silicon's unified memory makes `storageModeShared` the first target for
CPU emulator output and learner-visible rollout buffers. `storageModePrivate`
may be useful for purely GPU-local intermediate buffers, but the current Atari
layout intentionally avoids intermediates.

## Acceptance Tests

A native training backend is ready only when it passes:

- CPU reference parity on small synthetic frames
- MPS Python shader parity on Atari-shaped frames
- no-copy learner view shape and stride checks
- benchmark JSON with MPS memory counters
- repeated run stability without leaking command buffers or buffers

Until Torch MPS exposes a stable zero-copy bridge for external `MTLBuffer`
storage, the Python/MPS shader backend remains the verified Torch-training path.
