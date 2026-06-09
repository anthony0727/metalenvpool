import pytest
import torch

from metalenvpool import MetalRolloutBuffer, RolloutBufferConfig, RolloutObsLayout


def test_rollout_obs_layout_offsets_and_strides():
    layout = RolloutObsLayout(num_steps=3, num_envs=2, obs_shape=(4, 5, 6), dtype=torch.uint8)

    assert layout.rollout_obs_shape == (3, 2, 4, 5, 6)
    assert layout.current_obs_shape == (2, 4, 5, 6)
    assert layout.learner_batch_shape == (6, 4, 5, 6)
    assert layout.obs_strides == (30, 6, 1)
    assert layout.current_obs_strides == (120, 30, 6, 1)
    assert layout.rollout_obs_strides == (240, 120, 30, 6, 1)
    assert layout.current_obs_offset(1, 2, 3, 4) == 1 * 120 + 2 * 30 + 3 * 6 + 4
    assert layout.rollout_obs_offset(2, 1, 2, 3, 4) == 2 * 240 + 1 * 120 + 2 * 30 + 3 * 6 + 4
    assert layout.learner_batch_index(2, 1) == 5


def test_rollout_obs_layout_rejects_wrong_tensor_contract():
    layout = RolloutObsLayout(num_steps=2, num_envs=2, obs_shape=(4,), dtype=torch.float32)

    with pytest.raises(ValueError):
        layout.validate_rollout_obs(torch.empty((2, 2, 5)))
    with pytest.raises(TypeError):
        layout.validate_rollout_obs(torch.empty((2, 2, 4), dtype=torch.float64))
    with pytest.raises(ValueError):
        layout.validate_rollout_obs(torch.empty((2, 2, 4)).transpose(0, 1))


def test_rollout_buffer_learner_view_is_zero_copy():
    buf = MetalRolloutBuffer(
        RolloutBufferConfig(num_steps=2, num_envs=3, obs_shape=(4, 5, 6), obs_dtype=torch.uint8),
        device="cpu",
    )
    view = buf.learner_obs_view()

    assert tuple(view.shape) == (6, 4, 5, 6)
    assert view.data_ptr() == buf.obs.data_ptr()
