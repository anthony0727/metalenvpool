#include <metal_stdlib>
using namespace metal;

kernel void mpe_simple_step(
    device float* agent,
    device const float* action,
    device const float* landmark,
    device float* reward,
    device bool* terminated,
    device bool* truncated,
    device int* steps,
    constant uint& num_envs,
    constant float& dt,
    constant float& accel,
    constant float& damping,
    constant float& max_speed,
    constant int& max_cycles,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_envs) {
        return;
    }

    bool active = !(terminated[gid] || truncated[gid]);
    uint s = gid * 4;
    uint a = gid * 5;
    uint l = gid * 2;
    if (!active) {
        reward[gid] = 0.0f;
        return;
    }

    float left = clamp(action[a + 1], 0.0f, 1.0f);
    float right = clamp(action[a + 2], 0.0f, 1.0f);
    float down = clamp(action[a + 3], 0.0f, 1.0f);
    float up = clamp(action[a + 4], 0.0f, 1.0f);
    float fx = right - left;
    float fy = up - down;

    float vx = agent[s + 2] * (1.0f - damping) + fx * accel * dt;
    float vy = agent[s + 3] * (1.0f - damping) + fy * accel * dt;
    float speed = sqrt(vx * vx + vy * vy);
    if (speed > max_speed) {
        float scale = max_speed / max(speed, 1.0e-8f);
        vx *= scale;
        vy *= scale;
    }

    float x = agent[s] + vx * dt;
    float y = agent[s + 1] + vy * dt;
    agent[s] = x;
    agent[s + 1] = y;
    agent[s + 2] = vx;
    agent[s + 3] = vy;
    steps[gid] += 1;

    float dx = landmark[l] - x;
    float dy = landmark[l + 1] - y;
    float dist = sqrt(dx * dx + dy * dy);
    terminated[gid] = false;
    truncated[gid] = steps[gid] >= max_cycles;
    reward[gid] = -dist;
}
