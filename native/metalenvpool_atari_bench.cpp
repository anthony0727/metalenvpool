#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

struct Config {
    uint32_t num_envs = 16;
    uint32_t rollout_steps = 128;
    uint32_t iters = 1000;
    uint32_t warmup = 50;
    uint32_t sync_every = 0;
    uint32_t in_h = 210;
    uint32_t in_w = 160;
    uint32_t out_h = 84;
    uint32_t out_w = 84;
    uint32_t frame_stack = 4;
    uint32_t seed = 7;
    std::string metallib = "build/metalenvpool-native/atari_preprocess.metallib";
};

struct OwnedMetal {
    NS::AutoreleasePool* pool = nullptr;
    MTL::Device* device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* library = nullptr;
    MTL::Function* function = nullptr;
    MTL::ComputePipelineState* pipeline = nullptr;
    MTL::Buffer* frame_a = nullptr;
    MTL::Buffer* frame_b = nullptr;
    MTL::Buffer* obs = nullptr;
    MTL::Buffer* rollout_obs = nullptr;

    ~OwnedMetal() {
        if (rollout_obs) {
            rollout_obs->release();
        }
        if (obs) {
            obs->release();
        }
        if (frame_b) {
            frame_b->release();
        }
        if (frame_a) {
            frame_a->release();
        }
        if (pipeline) {
            pipeline->release();
        }
        if (function) {
            function->release();
        }
        if (library) {
            library->release();
        }
        if (queue) {
            queue->release();
        }
        if (device) {
            device->release();
        }
        if (pool) {
            pool->release();
        }
    }
};

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

std::string ns_error(NS::Error* error) {
    if (!error) {
        return "";
    }
    NS::String* desc = error->localizedDescription();
    if (!desc) {
        return "";
    }
    const char* utf8 = desc->utf8String();
    return utf8 ? std::string(utf8) : std::string();
}

uint32_t parse_u32(const char* value, const char* name) {
    char* end = nullptr;
    unsigned long out = std::strtoul(value, &end, 10);
    if (!value[0] || (end && *end) || out == 0 || out > UINT32_MAX) {
        die(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<uint32_t>(out);
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        auto need_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                die(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };
        std::string arg = argv[i];
        if (arg == "--num-envs") {
            cfg.num_envs = parse_u32(need_value("--num-envs"), "--num-envs");
        } else if (arg == "--rollout-steps") {
            cfg.rollout_steps = parse_u32(need_value("--rollout-steps"), "--rollout-steps");
        } else if (arg == "--iters") {
            cfg.iters = parse_u32(need_value("--iters"), "--iters");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_u32(need_value("--warmup"), "--warmup");
        } else if (arg == "--sync-every") {
            cfg.sync_every = parse_u32(need_value("--sync-every"), "--sync-every");
        } else if (arg == "--seed") {
            cfg.seed = parse_u32(need_value("--seed"), "--seed");
        } else if (arg == "--metallib") {
            cfg.metallib = need_value("--metallib");
        } else {
            die("unknown argument: " + arg);
        }
    }
    return cfg;
}

MTL::Buffer* new_shared_buffer(MTL::Device* device, size_t bytes, const char* name) {
    MTL::Buffer* buffer = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!buffer || !buffer->contents()) {
        die(std::string("failed to allocate Metal buffer: ") + name);
    }
    return buffer;
}

void fill_frames(MTL::Buffer* frame_a, MTL::Buffer* frame_b, size_t bytes, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    auto* a = static_cast<uint8_t*>(frame_a->contents());
    auto* b = static_cast<uint8_t*>(frame_b->contents());
    for (size_t i = 0; i < bytes; i++) {
        a[i] = static_cast<uint8_t>(dist(rng));
        b[i] = static_cast<uint8_t>(dist(rng));
    }
    frame_a->didModifyRange(NS::Range::Make(0, bytes));
    frame_b->didModifyRange(NS::Range::Make(0, bytes));
}

OwnedMetal make_backend(const Config& cfg, size_t frame_bytes, size_t obs_bytes, size_t rollout_bytes) {
    OwnedMetal m;
    m.pool = NS::AutoreleasePool::alloc()->init();
    m.device = MTL::CreateSystemDefaultDevice();
    if (!m.device) {
        die("MTL::CreateSystemDefaultDevice returned null");
    }
    m.queue = m.device->newCommandQueue();
    if (!m.queue) {
        die("failed to create MTLCommandQueue");
    }

    NS::Error* error = nullptr;
    NS::String* path = NS::String::string(cfg.metallib.c_str(), NS::UTF8StringEncoding);
    NS::URL* url = NS::URL::fileURLWithPath(path);
    m.library = m.device->newLibrary(url, &error);
    if (!m.library) {
        die("failed to load metallib: " + cfg.metallib + " " + ns_error(error));
    }
    NS::String* function_name = NS::String::string("atari_preprocess_write", NS::UTF8StringEncoding);
    m.function = m.library->newFunction(function_name);
    if (!m.function) {
        die("failed to find atari_preprocess_write in metallib");
    }
    m.pipeline = m.device->newComputePipelineState(m.function, &error);
    if (!m.pipeline) {
        die("failed to create compute pipeline: " + ns_error(error));
    }

    m.frame_a = new_shared_buffer(m.device, frame_bytes, "frame_a");
    m.frame_b = new_shared_buffer(m.device, frame_bytes, "frame_b");
    m.obs = new_shared_buffer(m.device, obs_bytes, "obs");
    m.rollout_obs = new_shared_buffer(m.device, rollout_bytes, "rollout_obs");
    std::memset(m.obs->contents(), 0, obs_bytes);
    std::memset(m.rollout_obs->contents(), 0, rollout_bytes);
    m.obs->didModifyRange(NS::Range::Make(0, obs_bytes));
    m.rollout_obs->didModifyRange(NS::Range::Make(0, rollout_bytes));
    return m;
}

MTL::CommandBuffer* run_step(OwnedMetal& m, const Config& cfg, uint32_t step_index, bool wait) {
    const uint32_t total_threads = cfg.num_envs * cfg.out_h * cfg.out_w;

    MTL::CommandBuffer* command_buffer = m.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = command_buffer->computeCommandEncoder();
    enc->setComputePipelineState(m.pipeline);
    enc->setBuffer(m.frame_a, 0, 0);
    enc->setBuffer(m.frame_b, 0, 1);
    enc->setBuffer(m.obs, 0, 2);
    enc->setBuffer(m.rollout_obs, 0, 3);
    enc->setBytes(&cfg.num_envs, sizeof(uint32_t), 4);
    enc->setBytes(&cfg.in_h, sizeof(uint32_t), 5);
    enc->setBytes(&cfg.in_w, sizeof(uint32_t), 6);
    enc->setBytes(&cfg.out_h, sizeof(uint32_t), 7);
    enc->setBytes(&cfg.out_w, sizeof(uint32_t), 8);
    enc->setBytes(&cfg.frame_stack, sizeof(uint32_t), 9);
    enc->setBytes(&step_index, sizeof(uint32_t), 10);

    const NS::UInteger width = std::max<NS::UInteger>(1, m.pipeline->threadExecutionWidth());
    const NS::UInteger max_threads = std::max<NS::UInteger>(1, m.pipeline->maxTotalThreadsPerThreadgroup());
    const NS::UInteger group_width = std::min<NS::UInteger>(width, max_threads);
    enc->dispatchThreads(MTL::Size(total_threads, 1, 1), MTL::Size(group_width, 1, 1));
    enc->endEncoding();
    command_buffer->retain();
    command_buffer->commit();
    if (wait) {
        command_buffer->waitUntilCompleted();
        command_buffer->release();
        return nullptr;
    }
    return command_buffer;
}

void release_after_wait(MTL::CommandBuffer* command_buffer) {
    if (!command_buffer) {
        return;
    }
    command_buffer->waitUntilCompleted();
    command_buffer->release();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        const size_t frame_bytes = static_cast<size_t>(cfg.num_envs) * cfg.in_h * cfg.in_w * 3;
        const size_t obs_bytes = static_cast<size_t>(cfg.num_envs) * cfg.frame_stack * cfg.out_h * cfg.out_w;
        const size_t rollout_bytes = static_cast<size_t>(cfg.rollout_steps) * obs_bytes;
        OwnedMetal m = make_backend(cfg, frame_bytes, obs_bytes, rollout_bytes);
        fill_frames(m.frame_a, m.frame_b, frame_bytes, cfg.seed);

        for (uint32_t i = 0; i < cfg.warmup; i++) {
            MTL::CommandBuffer* cb = run_step(m, cfg, i % cfg.rollout_steps, false);
            release_after_wait(cb);
        }

        const auto t0 = std::chrono::steady_clock::now();
        MTL::CommandBuffer* last = nullptr;
        for (uint32_t i = 0; i < cfg.iters; i++) {
            const bool wait = cfg.sync_every > 0 && ((i + 1) % cfg.sync_every == 0);
            MTL::CommandBuffer* cb = run_step(m, cfg, i % cfg.rollout_steps, wait);
            if (last) {
                last->release();
            }
            last = cb;
        }
        release_after_wait(last);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double stacked_obs = static_cast<double>(cfg.iters) * cfg.num_envs;
        const double raw_frames = stacked_obs * 2.0;

        std::cout << "{\n";
        std::cout << "  \"backend\": \"metal-cpp-atari-preprocess\",\n";
        std::cout << "  \"metallib\": \"" << cfg.metallib << "\",\n";
        std::cout << "  \"num_envs\": " << cfg.num_envs << ",\n";
        std::cout << "  \"rollout_steps\": " << cfg.rollout_steps << ",\n";
        std::cout << "  \"iters\": " << cfg.iters << ",\n";
        std::cout << "  \"warmup\": " << cfg.warmup << ",\n";
        std::cout << "  \"sync_every\": " << cfg.sync_every << ",\n";
        std::cout << "  \"seconds\": " << seconds << ",\n";
        std::cout << "  \"stacked_observations_per_second\": " << (stacked_obs / seconds) << ",\n";
        std::cout << "  \"raw_frames_per_second\": " << (raw_frames / seconds) << ",\n";
        std::cout << "  \"frame_bytes\": " << frame_bytes << ",\n";
        std::cout << "  \"obs_bytes\": " << obs_bytes << ",\n";
        std::cout << "  \"rollout_obs_bytes\": " << rollout_bytes << "\n";
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "metalenvpool_atari_bench: " << exc.what() << "\n";
        return 2;
    }
}
