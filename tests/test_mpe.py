import pytest
import torch

from metalenvpool import MetalMPESimplePool, MPESimpleConfig


def mpe_cfg(**kwargs):
    data = {"num_envs": 8, "max_cycles": 4}
    data.update(kwargs)
    return MPESimpleConfig(**data)


def test_mpe_simple_cpu_shapes():
    pool = MetalMPESimplePool(mpe_cfg(), device="cpu")

    obs = pool.reset(seed=3)
    assert tuple(obs.shape) == pool.obs_shape

    result = pool.step(torch.zeros(pool.action_shape))

    assert tuple(result.obs.shape) == pool.obs_shape
    assert tuple(result.reward.shape) == (pool.cfg.num_envs,)
    assert tuple(result.terminated.shape) == (pool.cfg.num_envs,)
    assert tuple(result.truncated.shape) == (pool.cfg.num_envs,)
    assert not pool.using_shader
    assert torch.isfinite(result.obs).all()
    assert torch.isfinite(result.reward).all()
    assert pool.single_action_spec.shape == (5,)
    assert pool.single_observation_spec.shape == (4,)


def test_mpe_simple_send_recv_matches_step_api():
    pool_a = MetalMPESimplePool(mpe_cfg(), device="cpu")
    pool_b = MetalMPESimplePool(mpe_cfg(), device="cpu")
    pool_a.reset(seed=5)
    pool_b.reset(seed=5)
    action = torch.rand(pool_a.action_shape)

    direct = pool_a.step(action)
    pool_b.send(action)
    split = pool_b.recv()

    torch.testing.assert_close(direct.obs, split.obs)
    torch.testing.assert_close(direct.reward, split.reward)
    assert torch.equal(direct.terminated, split.terminated)
    assert torch.equal(direct.truncated, split.truncated)


def test_mpe_simple_reset_done_resets_selected_slots_without_sync_query():
    pool = MetalMPESimplePool(mpe_cfg(num_envs=4), device="cpu")
    pool.reset(seed=17)
    old_agent = pool.agent.clone()
    old_landmark = pool.landmark.clone()
    pool.steps[:] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pool.truncated[:] = torch.tensor([False, True, False, True])
    done = torch.tensor([False, True, False, True])

    obs = pool.reset_done(done)

    assert tuple(obs.shape) == pool.obs_shape
    torch.testing.assert_close(pool.agent[0], old_agent[0])
    torch.testing.assert_close(pool.agent[2], old_agent[2])
    torch.testing.assert_close(pool.landmark[0], old_landmark[0])
    torch.testing.assert_close(pool.landmark[2], old_landmark[2])
    assert not torch.equal(pool.agent[1], old_agent[1])
    assert not torch.equal(pool.landmark[3], old_landmark[3])
    assert pool.steps.tolist() == [1, 0, 3, 0]
    assert pool.truncated.tolist() == [False, False, False, False]


def test_mpe_simple_truncates_at_max_cycles():
    pool = MetalMPESimplePool(mpe_cfg(num_envs=3, max_cycles=2), device="cpu")
    pool.reset(seed=19)

    first = pool.step(pool.zero_actions())
    first_truncated = first.truncated.clone()
    second = pool.step(pool.zero_actions())

    assert not first_truncated.any()
    assert second.truncated.all()
    assert not second.terminated.any()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mpe_simple_mps_shader_matches_torch_cpu_one_step():
    cfg = mpe_cfg()
    cpu = MetalMPESimplePool(cfg, device="cpu", use_shader=False)
    mps = MetalMPESimplePool(cfg, device="mps", use_shader=True)
    cpu.reset(seed=11)
    mps.agent[:] = cpu.agent.to("mps")
    mps.landmark[:] = cpu.landmark.to("mps")
    mps.reward[:] = cpu.reward.to("mps")
    mps.terminated[:] = cpu.terminated.to("mps")
    mps.truncated[:] = cpu.truncated.to("mps")
    mps.steps[:] = cpu.steps.to("mps")
    action = torch.linspace(0.0, 1.0, cfg.num_envs * 5, dtype=torch.float32).reshape(cfg.num_envs, 5)

    cpu_out = cpu.step(action)
    mps_out = mps.step(action.to("mps"))
    torch.mps.synchronize()

    assert mps.using_shader
    torch.testing.assert_close(cpu_out.obs, mps_out.obs.cpu(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cpu_out.reward, mps_out.reward.cpu(), rtol=1e-6, atol=1e-6)
    assert torch.equal(cpu_out.terminated, mps_out.terminated.cpu())
    assert torch.equal(cpu_out.truncated, mps_out.truncated.cpu())
