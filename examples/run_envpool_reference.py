"""Benchmark local EnvPool on the exact EnvPool paper task set.

The output is deliberately conservative: MetalEnvPool rows are marked partial
unless they include full environment execution for the same task.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/envpool_reference/2026-06-09"))
    parser.add_argument("--num-envs", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--steps-small", type=int, default=5_000)
    parser.add_argument("--steps-large", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    import_error = None
    try:
        import envpool
    except Exception as exc:  # pragma: no cover - exercised by artifact runner
        import_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "EnvPool import failed; on macOS the wheel may require Homebrew qt@5 for procgen linkage.",
        }
        envpool = None

    if envpool is not None:
        for num_envs in args.num_envs:
            steps = args.steps_small if num_envs == 1 else args.steps_large
            rows.append(bench_envpool_gym("Pong-v5", num_envs, steps, args.warmup, action_kind="pong"))
            rows.append(bench_envpool_gym("Ant-v3", num_envs, steps, args.warmup, action_kind="ant"))
            rows.append(bench_envpool_dm("CheetahRun-v1", num_envs, steps, args.warmup))

    metal_rows = load_metalenvpool_partial_rows()
    summary = {
        "source": "local EnvPool 1.2.5 reference on EnvPool paper task set",
        "envpool_import_error": import_error,
        "rows": rows,
        "metalenvpool_partial_rows": metal_rows,
        "win_assessment": assess_win(rows, metal_rows),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def bench_envpool_gym(env_id: str, num_envs: int, steps: int, warmup: int, *, action_kind: str) -> dict[str, Any]:
    import envpool

    env = envpool.make(env_id, env_type="gymnasium", num_envs=num_envs)
    env.reset()
    if action_kind == "pong":
        rng = np.random.default_rng(7)
        actions = rng.integers(0, 6, size=(steps, num_envs), dtype=np.int64)
    elif action_kind == "ant":
        actions = np.zeros((steps, num_envs, 8), dtype=np.float64)
    else:
        raise ValueError(action_kind)
    try:
        for i in range(warmup):
            env.step(actions[i % steps])
        start = time.perf_counter()
        for i in range(steps):
            env.step(actions[i])
        seconds = time.perf_counter() - start
    finally:
        env.close()
    return {
        "engine": "envpool",
        "task": env_id,
        "num_envs": num_envs,
        "steps_per_env": steps,
        "total_steps": steps * num_envs,
        "seconds": seconds,
        "steps_per_second": steps * num_envs / seconds,
        "full_env_execution": True,
    }


def bench_envpool_dm(env_id: str, num_envs: int, steps: int, warmup: int) -> dict[str, Any]:
    import envpool

    env = envpool.make_dm(env_id, num_envs=num_envs)
    env.reset()
    actions = np.zeros((steps, num_envs, 6), dtype=np.float64)
    try:
        for i in range(warmup):
            env.step(actions[i % steps])
        start = time.perf_counter()
        for i in range(steps):
            env.step(actions[i])
        seconds = time.perf_counter() - start
    finally:
        env.close()
    return {
        "engine": "envpool",
        "task": env_id,
        "num_envs": num_envs,
        "steps_per_env": steps,
        "total_steps": steps * num_envs,
        "seconds": seconds,
        "steps_per_second": steps * num_envs / seconds,
        "full_env_execution": True,
    }


def load_metalenvpool_partial_rows() -> list[dict[str, Any]]:
    path = Path("artifacts/envpool_paper_tasks/2026-06-09/pong.json")
    if not path.exists():
        return []
    pong = json.loads(path.read_text(encoding="utf-8"))
    devices = pong.get("devices", {})
    rows = []
    for device, metrics in sorted(devices.items()):
        rows.append(
            {
                "engine": "metalenvpool",
                "task": "Pong-v5",
                "device": device,
                "metric": "stacked_observations_per_second",
                "value": metrics.get("stacked_observations_per_second"),
                "full_env_execution": False,
                "note": "Atari preprocessing and rollout write only; ALE emulator execution is not included.",
            }
        )
    return rows


def assess_win(envpool_rows: list[dict[str, Any]], metal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not envpool_rows:
        return {
            "status": "not_reproduced",
            "reason": "No EnvPool reference rows were produced.",
        }
    if not any(row.get("full_env_execution") for row in metal_rows):
        return {
            "status": "not_won",
            "reason": (
                "MetalEnvPool has no full-environment execution row for Pong-v5, Ant-v3, or CheetahRun-v1. "
                "Partial preprocessing throughput cannot be compared as an EnvPool win."
            ),
        }
    return {
        "status": "needs_review",
        "reason": "At least one full-environment MetalEnvPool row exists; compare task-matched rows manually.",
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Local EnvPool Reference",
        "",
        "Exact EnvPool paper task set measured with local EnvPool where available.",
        "",
        "| Engine | Task | Num envs | Full env execution | Throughput |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['engine']} | `{row['task']}` | {row['num_envs']} | {row['full_env_execution']} | "
            f"{float(row['steps_per_second']):,.0f} steps/s |"
        )
    for row in summary["metalenvpool_partial_rows"]:
        lines.append(
            f"| {row['engine']} | `{row['task']}` | - | {row['full_env_execution']} | "
            f"{float(row['value']):,.0f} stacked obs/s |"
        )
    assessment = summary["win_assessment"]
    lines.extend(
        [
            "",
            f"Win assessment: **{assessment['status']}**.",
            "",
            assessment["reason"],
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
