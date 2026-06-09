"""Run the exact task names used in the EnvPool paper benchmark.

This is intentionally narrower than the broader hardware matrix. EnvPool's
isolated execution benchmark reports Atari Pong-v5, MuJoCo Ant-v3, and
dm_control cheetah run. MetalEnvPool does not yet implement all three dynamics;
this runner records that fact instead of silently substituting newer tasks.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metalenvpool import AtariPreprocessConfig, MetalAtariPreprocessor, memory_stats, synchronize


@dataclass(frozen=True)
class PaperTask:
    key: str
    paper_name: str
    local_name: str
    family: str
    paper_role: str


TASKS = {
    "pong": PaperTask(
        key="pong",
        paper_name="Atari Pong-v5",
        local_name="ALE/Pong-v5",
        family="atari",
        paper_role="isolated environment execution benchmark",
    ),
    "ant": PaperTask(
        key="ant",
        paper_name="MuJoCo Ant-v3",
        local_name="Ant-v3",
        family="mujoco-gym",
        paper_role="isolated environment execution benchmark",
    ),
    "cheetah_run": PaperTask(
        key="cheetah_run",
        paper_name="dm_control cheetah run",
        local_name="suite.load('cheetah', 'run')",
        family="dm_control",
        paper_role="single-environment speedup baseline",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/envpool_paper_tasks/2026-06-09"))
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--pong-iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--physics-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_key in args.tasks:
        task = TASKS[task_key]
        if task_key == "pong":
            result = run_pong(task, args)
        elif task_key == "ant":
            result = run_ant_v3(task, args)
        elif task_key == "cheetah_run":
            result = run_dm_control_cheetah_run(task, args)
        else:
            raise AssertionError(task_key)
        out_path = args.out_dir / f"{task_key}.json"
        result["artifact"] = str(out_path)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(result)

    summary = {
        "source": "EnvPool paper isolated benchmark task set",
        "tasks": {key: asdict(task) for key, task in TASKS.items()},
        "num_rows": len(rows),
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_pong(task: PaperTask, args: argparse.Namespace) -> dict[str, Any]:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    cfg = AtariPreprocessConfig(num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    frames_a, frames_b, action_space = capture_atari_frames(task.local_name, cfg.num_envs, args.seed)
    device_results = {}
    for device in available_torch_devices():
        pre = MetalAtariPreprocessor(cfg, device=device, use_shader=True)
        frame_a = frames_a.to(pre.device)
        frame_b = frames_b.to(pre.device)
        for i in range(args.warmup):
            pre.step(frame_a, frame_b, i % cfg.rollout_steps)
        synchronize(pre.device)
        start = time.perf_counter()
        for i in range(args.pong_iters):
            pre.step(frame_a, frame_b, i % cfg.rollout_steps)
        synchronize(pre.device)
        seconds = time.perf_counter() - start
        processed_obs = args.pong_iters * cfg.num_envs
        device_results[str(pre.device)] = {
            "seconds": seconds,
            "stacked_observations_per_second": processed_obs / seconds,
            "raw_frames_per_second": processed_obs * 2 / seconds,
            "using_shader": pre.using_shader,
            "memory": memory_stats(pre.device).__dict__,
        }

    return {
        "status": "ok",
        "task": asdict(task),
        "action_space": action_space,
        "coverage": {
            "metalenvpool_component": "Atari RGB preprocessing and learner-ready rollout write",
            "env_execution_included": False,
            "native_dynamics_implemented": False,
            "note": (
                "ALE emulator stepping is CPU/Gymnasium; MetalEnvPool times preprocessing/write for exact Pong frames."
            ),
        },
        "num_envs": cfg.num_envs,
        "rollout_steps": cfg.rollout_steps,
        "iters": args.pong_iters,
        "devices": device_results,
    }


def capture_atari_frames(env_id: str, num_envs: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    import gymnasium as gym

    frames_a = []
    frames_b = []
    action_space = ""
    envs = [gym.make(env_id, render_mode="rgb_array") for _ in range(num_envs)]
    try:
        for i, env in enumerate(envs):
            env.reset(seed=seed + i)
            frame_a = env.render()
            env.step(env.action_space.sample())
            frame_b = env.render()
            frames_a.append(frame_a)
            frames_b.append(frame_b)
        action_space = str(envs[0].action_space)
    finally:
        for env in envs:
            env.close()
    return (
        torch.as_tensor(np.stack(frames_a, axis=0), dtype=torch.uint8),
        torch.as_tensor(np.stack(frames_b, axis=0), dtype=torch.uint8),
        action_space,
    )


def run_ant_v3(task: PaperTask, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        env = gym.make(task.local_name)
    except Exception as exc:
        return blocked_result(
            task,
            exc,
            coverage_note=(
                "Exact Ant-v3 is a MuJoCo v2/v3 legacy gym task that currently requires deprecated mujoco-py "
                "on this stack. MetalEnvPool has no native Ant dynamics yet."
            ),
        )

    try:
        rng = np.random.default_rng(args.seed)
        obs, _info = env.reset(seed=args.seed)
        start = time.perf_counter()
        for _ in range(args.physics_steps):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(env.action_space.dtype)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                obs, _info = env.reset()
        seconds = time.perf_counter() - start
        return {
            "status": "ok",
            "task": asdict(task),
            "coverage": {
                "metalenvpool_component": None,
                "env_execution_included": True,
                "native_dynamics_implemented": False,
                "note": "CPU Gymnasium reference only; no MetalEnvPool Ant dynamics implementation yet.",
            },
            "steps": args.physics_steps,
            "seconds": seconds,
            "steps_per_second": args.physics_steps / seconds,
            "obs_shape": list(np.asarray(obs).shape),
            "action_space": str(env.action_space),
        }
    finally:
        env.close()


def run_dm_control_cheetah_run(task: PaperTask, args: argparse.Namespace) -> dict[str, Any]:
    try:
        from dm_control import suite

        env = suite.load("cheetah", "run", task_kwargs={"random": args.seed})
    except Exception as exc:
        return blocked_result(
            task,
            exc,
            coverage_note=(
                "Requires dm-control==1.0.38 with mujoco==3.6.0 for the EnvPool-docs-compatible stack. "
                "MetalEnvPool has no native dm_control cheetah dynamics yet."
            ),
        )

    rng = np.random.default_rng(args.seed)
    action_spec = env.action_spec()
    timestep = env.reset()
    start = time.perf_counter()
    for _ in range(args.physics_steps):
        action = rng.uniform(action_spec.minimum, action_spec.maximum).astype(action_spec.dtype)
        timestep = env.step(action)
        if timestep.last():
            timestep = env.reset()
    seconds = time.perf_counter() - start
    return {
        "status": "ok",
        "task": asdict(task),
        "coverage": {
            "metalenvpool_component": None,
            "env_execution_included": True,
            "native_dynamics_implemented": False,
            "note": "CPU dm_control reference only; no MetalEnvPool cheetah dynamics implementation yet.",
        },
        "steps": args.physics_steps,
        "seconds": seconds,
        "steps_per_second": args.physics_steps / seconds,
        "observation_keys": list(timestep.observation.keys()),
        "action_shape": list(action_spec.shape),
    }


def blocked_result(task: PaperTask, exc: Exception, *, coverage_note: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "task": asdict(task),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "coverage": {
            "metalenvpool_component": None,
            "env_execution_included": False,
            "native_dynamics_implemented": False,
            "note": coverage_note,
        },
    }


def available_torch_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    return devices


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EnvPool Paper Task Matrix",
        "",
        "Exact task names from the EnvPool isolated benchmark task set.",
        "",
        "| Paper task | Local id | Status | MetalEnvPool coverage | Throughput | Artifact |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in summary["rows"]:
        task = row["task"]
        throughput = throughput_text(row)
        coverage = row.get("coverage", {})
        component = coverage.get("metalenvpool_component") or "not implemented"
        lines.append(
            f"| {task['paper_name']} | `{task['local_name']}` | {row['status']} | {component} | "
            f"{throughput} | `{row['artifact']}` |"
        )
    return "\n".join(lines) + "\n"


def throughput_text(row: dict[str, Any]) -> str:
    if row["status"] != "ok":
        return "-"
    devices = row.get("devices")
    if isinstance(devices, dict):
        mps = devices.get("mps")
        cpu = devices.get("cpu")
        if isinstance(mps, dict) and isinstance(cpu, dict):
            return (
                f"MPS {float(mps['stacked_observations_per_second']):,.0f} stacked obs/s; "
                f"CPU {float(cpu['stacked_observations_per_second']):,.0f}"
            )
    if "steps_per_second" in row:
        return f"{float(row['steps_per_second']):,.0f} steps/s"
    return "-"


if __name__ == "__main__":
    main()
