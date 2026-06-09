import numpy as np
import torch

from metalenvpool import (
    MetalMPESimplePool,
    MetalPointPool,
    MPESimpleConfig,
    PointConfig,
    TensorEnv,
    TensorSpec,
    check_tensor_env,
)


def test_tensor_spec_maps_to_gymnasium_space():
    spec = TensorSpec(shape=(2,), dtype=torch.float32, low=-1.0, high=1.0)
    space = spec.gymnasium_space()

    assert space.shape == (2,)
    assert space.dtype == np.float32


def test_point_pool_implements_tensor_env_contract():
    env = MetalPointPool(PointConfig(num_envs=8, max_steps=4), device="cpu")
    out = check_tensor_env(env, seed=3)

    assert isinstance(env, TensorEnv)
    assert out["num_envs"] == 8
    assert env.single_observation_space.shape == (6,)
    assert env.single_action_space.shape == (2,)
    assert env.observation_space.shape == (8, 6)
    assert env.action_space.shape == (8, 2)


def test_mpe_simple_pool_implements_tensor_env_contract():
    env = MetalMPESimplePool(MPESimpleConfig(num_envs=8, max_cycles=4), device="cpu")
    out = check_tensor_env(env, seed=3)

    assert isinstance(env, TensorEnv)
    assert out["num_envs"] == 8
    assert env.single_observation_space.shape == (4,)
    assert env.single_action_space.shape == (5,)
    assert env.observation_space.shape == (8, 4)
    assert env.action_space.shape == (8, 5)
