#include <metal_stdlib>
using namespace metal;

kernel void point_step(
    device float* state,
    device const float* action,
    device const float* target,
    device float* reward,
    device bool* terminated,
    device bool* truncated,
    device int* steps,
    constant uint& num_envs,
    constant float& dt,
    constant float& max_speed,
    constant float& max_accel,
    constant float& drag,
    constant float& world_size,
    constant float& success_radius,
    constant int& max_steps,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_envs) {
        return;
    }
    bool active = !(terminated[gid] || truncated[gid]);
    uint s = gid * 4;
    uint a = gid * 2;
    if (!active) {
        reward[gid] = 0.0f;
        return;
    }

    float ax = clamp(action[a], -max_accel, max_accel);
    float ay = clamp(action[a + 1], -max_accel, max_accel);
    float vx = state[s + 2] * (1.0f - drag) + ax * dt;
    float vy = state[s + 3] * (1.0f - drag) + ay * dt;
    float speed = sqrt(vx * vx + vy * vy);
    if (speed > max_speed) {
        float scale = max_speed / max(speed, 1.0e-8f);
        vx *= scale;
        vy *= scale;
    }

    float x = state[s] + vx * dt;
    float y = state[s + 1] + vy * dt;
    if (x < 0.0f) {
        x = 0.0f;
        vx *= -0.5f;
    } else if (x > world_size) {
        x = world_size;
        vx *= -0.5f;
    }
    if (y < 0.0f) {
        y = 0.0f;
        vy *= -0.5f;
    } else if (y > world_size) {
        y = world_size;
        vy *= -0.5f;
    }

    state[s] = x;
    state[s + 1] = y;
    state[s + 2] = vx;
    state[s + 3] = vy;
    steps[gid] += 1;

    float dx = target[a] - x;
    float dy = target[a + 1] - y;
    float dist = sqrt(dx * dx + dy * dy);
    bool hit = dist <= success_radius;
    terminated[gid] = hit;
    truncated[gid] = (steps[gid] >= max_steps) && !hit;
    reward[gid] = -dist - 0.001f + (hit ? 1.0f : 0.0f);
}
