#include <metal_stdlib>
using namespace metal;

kernel void atari_preprocess_write(
    device const uchar* frame_a,
    device const uchar* frame_b,
    device uchar* obs,
    device uchar* rollout_obs,
    constant uint& num_envs,
    constant uint& in_h,
    constant uint& in_w,
    constant uint& out_h,
    constant uint& out_w,
    constant uint& frame_stack,
    constant uint& step_index,
    uint gid [[thread_position_in_grid]]
) {
    uint pixels = out_h * out_w;
    uint total = num_envs * pixels;
    if (gid >= total) {
        return;
    }

    uint n = gid / pixels;
    uint p = gid - n * pixels;
    uint oy = p / out_w;
    uint ox = p - oy * out_w;
    uint sy = (oy * in_h) / out_h;
    uint sx = (ox * in_w) / out_w;
    uint src_base = ((n * in_h + sy) * in_w + sx) * 3;

    uchar r = max(frame_a[src_base], frame_b[src_base]);
    uchar g = max(frame_a[src_base + 1], frame_b[src_base + 1]);
    uchar b = max(frame_a[src_base + 2], frame_b[src_base + 2]);
    uchar gray = uchar((uint(77) * uint(r) + uint(150) * uint(g) + uint(29) * uint(b)) >> 8);

    uint obs_base = (n * frame_stack * pixels) + p;
    for (uint c = 0; c + 1 < frame_stack; c++) {
        obs[obs_base + c * pixels] = obs[obs_base + (c + 1) * pixels];
    }
    obs[obs_base + (frame_stack - 1) * pixels] = gray;

    uint rollout_base = ((step_index * num_envs + n) * frame_stack * pixels) + p;
    for (uint c = 0; c < frame_stack; c++) {
        rollout_obs[rollout_base + c * pixels] = obs[obs_base + c * pixels];
    }
}
