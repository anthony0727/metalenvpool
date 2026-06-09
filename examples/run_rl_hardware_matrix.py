"""Run and summarize fair SB3/Gymnasium hardware controls.

This script is intentionally separate from MetalEnvPool custom-task benchmarks.
It answers: for exact public Gymnasium tasks, how do the Apple M4 SoC CPU
backend, Apple M4 SoC MPS GPU backend, Colab CUDA GPU, and Colab TPU-runtime
controls compare under the same SB3 PPO script?
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskPreset:
    key: str
    env_id: str
    family: str
    num_envs: int
    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int


TASKS = {
    "cartpole": TaskPreset(
        key="cartpole",
        env_id="CartPole-v1",
        family="classic-control-discrete",
        num_envs=16,
        total_timesteps=50_000,
        n_steps=128,
        batch_size=512,
        n_epochs=4,
    ),
    "pendulum": TaskPreset(
        key="pendulum",
        env_id="Pendulum-v1",
        family="classic-control-continuous",
        num_envs=16,
        total_timesteps=50_000,
        n_steps=128,
        batch_size=512,
        n_epochs=4,
    ),
    "idp": TaskPreset(
        key="idp",
        env_id="InvertedDoublePendulum-v5",
        family="mujoco-continuous",
        num_envs=8,
        total_timesteps=50_000,
        n_steps=512,
        batch_size=1024,
        n_epochs=10,
    ),
}

LOCAL_RUNNERS = {
    "mac-cpu": {
        "label": "Apple M4 SoC CPU backend",
        "accelerator_class": "apple-silicon-cpu",
        "device": "cpu",
    },
    "mac-mps": {
        "label": "Apple M4 SoC MPS GPU backend",
        "accelerator_class": "apple-silicon-gpu",
        "device": "mps",
    },
}

COLAB_RUNNERS = {
    "t4": {
        "label": "Colab Tesla T4",
        "accelerator_class": "cuda-gpu",
        "colab_args": ["--gpu", "T4"],
        "requested_device": "auto",
    },
    "l4": {
        "label": "Colab NVIDIA L4",
        "accelerator_class": "cuda-gpu",
        "colab_args": ["--gpu", "L4"],
        "requested_device": "auto",
    },
    "tpu-v5e1": {
        "label": "Colab TPU v5e1 runtime",
        "accelerator_class": "tpu-runtime",
        "colab_args": ["--tpu", "v5e1"],
        "requested_device": "auto",
    },
}

TASK_ORDER = list(TASKS)
RUNNER_ORDER = ["mac-cpu", "mac-mps", "t4", "l4", "tpu-v5e1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/rl_hardware_matrix/2026-06-09"))
    parser.add_argument("--tasks", nargs="+", choices=TASK_ORDER, default=TASK_ORDER)
    parser.add_argument("--run-local", action="store_true")
    parser.add_argument("--run-colab", nargs="+", choices=list(COLAB_RUNNERS), default=[])
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        selected_tasks = [TASKS[key] for key in args.tasks]
        if args.run_local:
            for task in selected_tasks:
                for runner_key in LOCAL_RUNNERS:
                    run_local(task, runner_key, args.out_dir, args.eval_episodes, args.reuse_existing)
        for runner_key in args.run_colab:
            for task in selected_tasks:
                run_colab(task, runner_key, args.out_dir, args.eval_episodes, args.reuse_existing)

    runs = load_runs(args.out_dir)
    summary = summarize(runs)
    (args.out_dir / "hardware_matrix.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, args.out_dir / "hardware_matrix.md")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_local(
    task: TaskPreset,
    runner_key: str,
    out_dir: Path,
    eval_episodes: int,
    reuse_existing: bool,
) -> None:
    runner = LOCAL_RUNNERS[runner_key]
    out_path = run_path(out_dir, task.key, runner_key)
    if reuse_existing and out_path.exists():
        return
    cmd = [
        sys.executable,
        "examples/colab_sb3_gymnasium_benchmark.py",
        *task_args(task),
        "--device",
        runner["device"],
        "--eval-episodes",
        str(eval_episodes),
        "--no-install",
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"local run failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    add_matrix_metadata(data, task, runner_key, runner["label"], runner["accelerator_class"], cmd)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_colab(
    task: TaskPreset,
    runner_key: str,
    out_dir: Path,
    eval_episodes: int,
    reuse_existing: bool,
) -> None:
    runner = COLAB_RUNNERS[runner_key]
    out_path = run_path(out_dir, task.key, runner_key)
    if reuse_existing and out_path.exists():
        return
    cmd = [
        "colab",
        "run",
        *runner["colab_args"],
        "examples/colab_sb3_gymnasium_benchmark.py",
        *task_args(task),
        "--device",
        runner["requested_device"],
        "--eval-episodes",
        str(eval_episodes),
        "--out",
        "colab_sb3_gymnasium_result.json",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        try:
            data = parse_first_json_object(proc.stdout)
        except ValueError:
            data = {}
        if data:
            data["transport_warning"] = {
                "returncode": proc.returncode,
                "reason": "colab CLI returned nonzero after emitting a completed JSON result",
            }
            add_matrix_metadata(data, task, runner_key, runner["label"], runner["accelerator_class"], cmd)
            out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        failure = {
            "status": "failed",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
        add_matrix_metadata(failure, task, runner_key, runner["label"], runner["accelerator_class"], cmd)
        out_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"colab run failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    data = parse_first_json_object(proc.stdout)
    add_matrix_metadata(data, task, runner_key, runner["label"], runner["accelerator_class"], cmd)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_args(task: TaskPreset) -> list[str]:
    return [
        "--env-id",
        task.env_id,
        "--num-envs",
        str(task.num_envs),
        "--total-timesteps",
        str(task.total_timesteps),
        "--vec-env",
        "dummy",
        "--seed",
        "7",
        "--n-steps",
        str(task.n_steps),
        "--batch-size",
        str(task.batch_size),
        "--n-epochs",
        str(task.n_epochs),
    ]


def run_path(out_dir: Path, task_key: str, runner_key: str) -> Path:
    return out_dir / f"{task_key}_{runner_key}.json"


def add_matrix_metadata(
    data: dict[str, Any],
    task: TaskPreset,
    runner_key: str,
    runner_label: str,
    accelerator_class: str,
    cmd: list[str],
) -> None:
    data["matrix"] = {
        "task_key": task.key,
        "task_family": task.family,
        "runner_key": runner_key,
        "runner_label": runner_label,
        "accelerator_class": accelerator_class,
        "command": cmd,
    }


def parse_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"no JSON object found in output:\n{text}")


def load_runs(out_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "hardware_matrix.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data = recover_completed_colab_json(data)
        if not is_task_matched_current_result(data):
            continue
        data["_artifact"] = str(path)
        if "matrix" in data:
            runs.append(data)
    return runs


def is_task_matched_current_result(data: dict[str, Any]) -> bool:
    matrix = data.get("matrix")
    if not isinstance(matrix, dict):
        return False
    task = TASKS.get(matrix.get("task_key"))
    if task is None:
        return False
    if data.get("status", "ok") != "ok":
        return False
    return data.get("total_timesteps") == task.total_timesteps


def recover_completed_colab_json(data: dict[str, Any]) -> dict[str, Any]:
    """Recover a completed Colab result when the CLI timed out after stdout JSON."""

    if data.get("status") != "failed" or "stdout" not in data or "matrix" not in data:
        return data
    try:
        recovered = parse_first_json_object(str(data["stdout"]))
    except ValueError:
        return data
    recovered["matrix"] = data["matrix"]
    recovered["transport_warning"] = {
        "returncode": data.get("returncode"),
        "reason": "colab CLI returned nonzero after emitting a completed JSON result",
    }
    return recovered


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_runs": len(runs),
        "tasks": {key: task.__dict__ for key, task in TASKS.items()},
        "rows": sorted((row_for_run(run) for run in runs), key=row_sort_key),
    }


def row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    matrix = run["matrix"]
    accelerator = run.get("accelerator", {})
    return {
        "task": matrix["task_key"],
        "env_id": run.get("env_id"),
        "family": matrix["task_family"],
        "runner": runner_label_for(matrix["runner_key"]),
        "runner_key": matrix["runner_key"],
        "accelerator_class": matrix["accelerator_class"],
        "resolved_device": run.get("resolved_device"),
        "model_device": run.get("model_device"),
        "cuda_devices": accelerator.get("cuda_devices"),
        "torch_xla_available": accelerator.get("torch_xla_available"),
        "xla_device": accelerator.get("xla_device"),
        "total_timesteps": run.get("total_timesteps"),
        "seconds": run.get("seconds"),
        "env_steps_per_second": run.get("env_steps_per_second"),
        "eval_mean_episode_return": run.get("eval_mean_episode_return"),
        "eval_std_episode_return": run.get("eval_std_episode_return"),
        "artifact": run.get("_artifact"),
        "status": run.get("status", "ok"),
        "note": run.get("note", ""),
    }


def runner_label_for(runner_key: str) -> str:
    if runner_key in LOCAL_RUNNERS:
        return str(LOCAL_RUNNERS[runner_key]["label"])
    if runner_key in COLAB_RUNNERS:
        return str(COLAB_RUNNERS[runner_key]["label"])
    return runner_key


def row_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    task_index = TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else len(TASK_ORDER)
    runner_key = row["runner_key"]
    runner_index = RUNNER_ORDER.index(runner_key) if runner_key in RUNNER_ORDER else len(RUNNER_ORDER)
    return task_index, runner_index


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# RL Hardware Matrix",
        "",
        "Exact public Gymnasium tasks using the same Stable-Baselines3 PPO runner.",
        "These rows compare framework/hardware behavior, not a MetalEnvPool tensor-native task win.",
        "",
        "| Task | Runner | Device | Timesteps | Steps/s | Eval return | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        notes = note_for_row(row)
        lines.append(
            f"| `{row['env_id']}` | {row['runner']} | `{row['model_device']}` | "
            f"{int_or_dash(row['total_timesteps'])} | {float_or_dash(row['env_steps_per_second'])} | "
            f"{float_or_dash(row['eval_mean_episode_return'])} | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def note_for_row(row: dict[str, Any]) -> str:
    if row["runner_key"] == "tpu-v5e1" and row["model_device"] == "cpu":
        return "TPU runtime; SB3 resolved to CPU fallback"
    if row["runner_key"] in {"mac-mps", "t4"}:
        return "MLP PPO GPU control"
    return "CPU control"


def int_or_dash(value: Any) -> str:
    return "-" if value is None else f"{int(value):,}"


def float_or_dash(value: Any) -> str:
    return "-" if value is None else f"{float(value):,.1f}"


if __name__ == "__main__":
    main()
