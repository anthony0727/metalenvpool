import torch

from metalenvpool import MetalRolloutBuffer, RolloutBufferConfig


def test_rollout_buffer_shapes_cpu():
    buf = MetalRolloutBuffer(
        RolloutBufferConfig(num_steps=8, num_envs=4, obs_shape=(4, 84, 84), action_shape=(), obs_dtype=torch.uint8),
        device="cpu",
    )

    assert tuple(buf.obs.shape) == (8, 4, 4, 84, 84)
    assert tuple(buf.actions.shape) == (8, 4)
    assert tuple(buf.rewards.shape) == (8, 4)
    assert buf.obs.dtype == torch.uint8

    buf.zero_()
    assert int(buf.obs.sum()) == 0
