import pytest
import torch

from metalenvpool import AtariPreprocessConfig, MetalAtariPreprocessor


def small_cfg(**kwargs):
    data = {
        "num_envs": 2,
        "rollout_steps": 3,
        "in_height": 10,
        "in_width": 8,
        "out_height": 5,
        "out_width": 4,
        "frame_stack": 4,
    }
    data.update(kwargs)
    return AtariPreprocessConfig(**data)


def test_atari_preprocess_cpu_reference_shapes():
    cfg = small_cfg()
    pre = MetalAtariPreprocessor(cfg, device="cpu")
    a = torch.arange(cfg.num_envs * cfg.in_height * cfg.in_width * 3, dtype=torch.uint8).reshape(
        cfg.num_envs,
        cfg.in_height,
        cfg.in_width,
        3,
    )
    b = torch.flip(a, dims=[1])

    out0 = pre.step(a, b, 0)
    out1 = pre.step(a, b, 1)

    assert tuple(out0.shape) == (cfg.num_envs, cfg.frame_stack, cfg.out_height, cfg.out_width)
    assert tuple(pre.rollout.obs.shape) == (
        cfg.rollout_steps,
        cfg.num_envs,
        cfg.frame_stack,
        cfg.out_height,
        cfg.out_width,
    )
    assert tuple(pre.rollout.learner_obs_view().shape) == (
        cfg.rollout_steps * cfg.num_envs,
        cfg.frame_stack,
        cfg.out_height,
        cfg.out_width,
    )
    assert out0.dtype == torch.uint8
    torch.testing.assert_close(out0[:, -1], out1[:, -2])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_atari_preprocess_mps_shader_matches_cpu_reference():
    cfg = small_cfg()
    cpu = MetalAtariPreprocessor(cfg, device="cpu", use_shader=False)
    mps = MetalAtariPreprocessor(cfg, device="mps", use_shader=True)
    a = torch.randint(0, 256, (cfg.num_envs, cfg.in_height, cfg.in_width, 3), dtype=torch.uint8)
    b = torch.randint(0, 256, (cfg.num_envs, cfg.in_height, cfg.in_width, 3), dtype=torch.uint8)

    cpu0 = cpu.step(a, b, 0)
    mps0 = mps.step(a.to("mps"), b.to("mps"), 0)
    cpu1 = cpu.step(b, a, 1)
    mps1 = mps.step(b.to("mps"), a.to("mps"), 1)
    torch.mps.synchronize()

    assert mps.using_shader
    torch.testing.assert_close(cpu0, mps0.cpu())
    torch.testing.assert_close(cpu1, mps1.cpu())
