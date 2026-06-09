from metalenvpool.native import shader_source


def test_packaged_metal_shader_sources_are_available():
    atari = shader_source("atari_preprocess.metal")
    point = shader_source("point_step.metal")

    assert "kernel void atari_preprocess_write" in atari
    assert "device uchar* rollout_obs" in atari
    assert "kernel void point_step" in point
    assert "device float* state" in point
