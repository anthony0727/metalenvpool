import pytest
import torch

from metalenvpool import MetalPointPool, PointConfig, step_with_autoreset


def point_cfg(**kwargs):
    data = {"num_envs": 8, "max_steps": 4}
    data.update(kwargs)
    return PointConfig(**data)


def test_point_cpu_shapes():
    pool = MetalPointPool(point_cfg(), device="cpu")

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


def test_point_send_recv_matches_step_api():
    pool_a = MetalPointPool(point_cfg(), device="cpu")
    pool_b = MetalPointPool(point_cfg(), device="cpu")
    pool_a.reset(seed=5)
    pool_b.reset(seed=5)
    action = torch.zeros(pool_a.action_shape)

    direct = pool_a.step(action)
    pool_b.send(action)
    split = pool_b.recv()

    torch.testing.assert_close(direct.obs, split.obs)
    torch.testing.assert_close(direct.reward, split.reward)
    assert torch.equal(direct.terminated, split.terminated)


def test_point_reset_done_resets_selected_slots_without_sync_query():
    pool = MetalPointPool(point_cfg(num_envs=4), device="cpu")
    pool.reset(seed=17)
    old_state = pool.state.clone()
    old_target = pool.target.clone()
    pool.steps[:] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pool.terminated[:] = torch.tensor([False, True, False, True])
    done = torch.tensor([False, True, False, True])

    obs = pool.reset_done(done)

    assert tuple(obs.shape) == pool.obs_shape
    torch.testing.assert_close(pool.state[0], old_state[0])
    torch.testing.assert_close(pool.state[2], old_state[2])
    torch.testing.assert_close(pool.target[0], old_target[0])
    torch.testing.assert_close(pool.target[2], old_target[2])
    assert not torch.equal(pool.state[1], old_state[1])
    assert not torch.equal(pool.target[3], old_target[3])
    assert pool.steps.tolist() == [1, 0, 3, 0]
    assert pool.terminated.tolist() == [False, False, False, False]


def test_benchmark_step_keeps_environments_active_after_episode_limit():
    pool = MetalPointPool(point_cfg(num_envs=4, max_steps=2), device="cpu", use_shader=False)
    pool.reset(seed=23)
    action = torch.zeros(pool.action_shape)

    first = step_with_autoreset(pool, action)
    completed = step_with_autoreset(pool, action)
    for _ in range(4):
        step_with_autoreset(pool, action)

    assert not first.truncated.any()
    assert completed.truncated.all()
    assert completed.info["steps"].tolist() == [pool.cfg.max_steps] * pool.cfg.num_envs
    assert not pool.terminated.any()
    assert not pool.truncated.any()
    assert max(pool.steps.tolist()) < pool.cfg.max_steps


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_point_mps_shader_matches_torch_cpu_one_step():
    cfg = point_cfg()
    cpu = MetalPointPool(cfg, device="cpu", use_shader=False)
    mps = MetalPointPool(cfg, device="mps", use_shader=True)
    cpu.reset(seed=11)
    mps.state[:] = cpu.state.to("mps")
    mps.target[:] = cpu.target.to("mps")
    mps.reward[:] = cpu.reward.to("mps")
    mps.terminated[:] = cpu.terminated.to("mps")
    mps.truncated[:] = cpu.truncated.to("mps")
    mps.steps[:] = cpu.steps.to("mps")
    action = torch.linspace(-1.0, 1.0, cfg.num_envs * 2, dtype=torch.float32).reshape(cfg.num_envs, 2)

    cpu_out = cpu.step(action)
    mps_out = mps.step(action.to("mps"))
    torch.mps.synchronize()

    assert mps.using_shader
    torch.testing.assert_close(cpu_out.obs, mps_out.obs.cpu(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cpu_out.reward, mps_out.reward.cpu(), rtol=1e-6, atol=1e-6)
    assert torch.equal(cpu_out.terminated, mps_out.terminated.cpu())
    assert torch.equal(cpu_out.truncated, mps_out.truncated.cpu())
