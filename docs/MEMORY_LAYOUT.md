# Memory Layout ABI

MetalEnvPool's main Apple Silicon bet is a learner-ready rollout layout, not a
generic GPU wrapper around Gymnasium.

## Rollout Observation ABI

Observations are stored as contiguous row-major tensors:

```text
[T, N, *obs_shape]
```

For Atari:

```text
[rollout_steps, num_envs, frame_stack, 84, 84] uint8
```

The learner reads the same storage as:

```text
[T * N, frame_stack, 84, 84]
```

That view is zero-copy when the buffer is contiguous. The environment kernel
writes exactly where PPO/DQN reads, instead of producing Python objects or
intermediate arrays that later need copying, stacking, or transposing.

## Strides

The ABI is defined by `RolloutObsLayout`:

```text
obs_strides         = row_major(*obs_shape)
current_obs_strides = [prod(obs_shape), *obs_strides]
rollout_obs_strides = [N * prod(obs_shape), prod(obs_shape), *obs_strides]
```

For Atari `[T, N, C, H, W]`:

```text
rollout offset =
  t * (N * C * H * W) +
  n * (C * H * W) +
  c * (H * W) +
  y * W +
  x
```

This is the contract the Metal shader, Python CPU fallback, benchmark code, and
learner view all share.

## Fused Atari Write

`native/atari_preprocess.metal` performs these stages in one kernel:

1. max-pool two raw RGB frames for flicker reduction
2. nearest-neighbor resize from `210x160` to `84x84`
3. integer grayscale conversion
4. frame-stack shift in current observation storage
5. direct write into rollout observation storage

The output is already learner-shaped. On Apple SoCs, this matters because CPU
and GPU share memory, and Metal can write into the exact buffer layout the
learner will consume.

## What Is Not Claimed Yet

This layout does not make the ALE emulator itself run on Metal. Current Atari
benchmarks isolate preprocessing and rollout writes using real ALE frames.
Full RL training wins require a matched PPO/DQN runner with continuous emulator
stepping, policy inference, optimizer work, and reward curves.
