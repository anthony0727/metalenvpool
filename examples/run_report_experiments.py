"""Run repeated benchmark experiments for the MetalEnvPool technical report."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (7, 11, 13)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/report_runs/latest"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--point-steps", type=int, default=2000)
    parser.add_argument("--atari-iters", type=int, default=500)
    parser.add_argument("--mpe-timesteps", type=int, default=2_097_152)
    parser.add_argument("--sb3-timesteps", type=int, default=50_000)
    parser.add_argument("--skip-sb3", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate summary.md and figures from summary.json.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing summary for --plot-only: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        write_markdown_summary(summary, out_dir / "summary.md")
        write_figures(summary, out_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    raw_path = out_dir / "raw_runs.jsonl"
    raw_path.write_text("", encoding="utf-8")

    runs: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for seed in args.seeds:
        for device in ("cpu", "mps"):
            for action_mode in ("zero", "random"):
                runs.append(
                    run_json(
                        "point_step",
                        [
                            "examples/bench_point.py",
                            "--device",
                            device,
                            "--action-mode",
                            action_mode,
                            "--num-envs",
                            "65536",
                            "--steps",
                            str(args.point_steps),
                            "--warmup",
                            "100",
                            "--seed",
                            str(seed),
                        ],
                        raw_path,
                    )
                )

        for device in ("cpu", "mps"):
            runs.append(
                run_json(
                    "atari_preprocess_synthetic",
                    [
                        "examples/bench_atari_pipeline.py",
                        "--device",
                        device,
                        "--source",
                        "synthetic",
                        "--num-envs",
                        "16",
                        "--rollout-steps",
                        "128",
                        "--iters",
                        str(args.atari_iters),
                        "--warmup",
                        "50",
                        "--seed",
                        str(seed),
                    ],
                    raw_path,
                )
            )

        if not args.skip_ppo:
            runs.append(
                run_json(
                    "mpe_simple_ppo",
                    [
                        "examples/train_mpe_simple_ppo.py",
                        "--device",
                        "auto",
                        "--seed",
                        str(seed),
                        "--num-envs",
                        "8192",
                        "--num-steps",
                        "64",
                        "--total-timesteps",
                        str(args.mpe_timesteps),
                        "--num-minibatches",
                        "8",
                        "--update-epochs",
                        "2",
                    ],
                    raw_path,
                )
            )

        if not args.skip_sb3:
            for device in ("cpu", "mps"):
                runs.append(
                    run_json(
                        "sb3_idp",
                        [
                            "examples/colab_sb3_gymnasium_benchmark.py",
                            "--env-id",
                            "InvertedDoublePendulum-v5",
                            "--num-envs",
                            "8",
                            "--total-timesteps",
                            str(args.sb3_timesteps),
                            "--device",
                            device,
                            "--vec-env",
                            "dummy",
                            "--seed",
                            str(seed),
                            "--n-steps",
                            "512",
                            "--batch-size",
                            "1024",
                            "--n-epochs",
                            "10",
                            "--no-install",
                            "--out",
                            str(out_dir / f"sb3_idp_{device}_seed{seed}.json"),
                        ],
                        raw_path,
                    )
                )

    summary = summarize(runs, started_at=started_at, seeds=args.seeds)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(summary, out_dir / "summary.md")
    write_figures(summary, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_json(kind: str, argv: list[str], raw_path: Path) -> dict[str, Any]:
    cmd = [sys.executable, *argv]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        payload = {
            "kind": kind,
            "status": "failed",
            "cmd": cmd,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        append_jsonl(raw_path, payload)
        raise RuntimeError(f"{kind} failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    data = parse_last_json(proc.stdout)
    data["kind"] = kind
    data["cmd"] = cmd
    data["elapsed_seconds"] = elapsed
    data["status"] = "ok"
    append_jsonl(raw_path, data)
    return data


def parse_last_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object found in output:\n{text}")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"JSON output is not an object:\n{text}")
    return obj


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize(runs: list[dict[str, Any]], *, started_at: str, seeds: list[int]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[group_key(run)].append(run)
    return {
        "started_at": started_at,
        "seeds": seeds,
        "num_runs": len(runs),
        "groups": {key: summarize_group(key, rows) for key, rows in sorted(groups.items())},
    }


def group_key(run: dict[str, Any]) -> str:
    kind = run["kind"]
    if kind == "point_step":
        return f"point/{run['device']}/{run['action_mode']}"
    if kind == "atari_preprocess_synthetic":
        return f"atari_synth/{run['device']}"
    if kind == "mpe_simple_ppo":
        return f"mpe_ppo/{run['device']}"
    if kind == "sb3_idp":
        return f"sb3_idp/{run['model_device']}"
    return kind


def summarize_group(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_name = metric_for_key(key)
    values = [float(row[metric_name]) for row in rows]
    n = len(values)
    sd = stdev(values) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    out: dict[str, Any] = {
        "metric": metric_name,
        "n": n,
        "mean": mean(values),
        "std": sd,
        "sem": sem,
        "ci95": 1.96 * sem,
        "min": min(values),
        "max": max(values),
        "values": values,
    }
    if key.startswith("mpe_ppo/"):
        out["first_mean_rollout_reward"] = stat([float(row["first_mean_rollout_reward"]) for row in rows])
        out["final_mean_rollout_reward"] = stat([float(row["final_mean_rollout_reward"]) for row in rows])
        out["eval_mean_episode_return"] = stat([float(row["eval_mean_episode_return"]) for row in rows])
    return out


def metric_for_key(key: str) -> str:
    if key.startswith("atari_synth/"):
        return "stacked_observations_per_second"
    return "env_steps_per_second"


def stat(values: list[float]) -> dict[str, float]:
    n = len(values)
    sd = stdev(values) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean(values),
        "std": sd,
        "sem": sem,
        "ci95": 1.96 * sem,
        "min": min(values),
        "max": max(values),
    }


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Report Experiment Summary",
        "",
        f"- started_at: `{summary['started_at']}`",
        f"- seeds: `{summary['seeds']}`",
        f"- runs: `{summary['num_runs']}`",
        "",
        "| Group | n | Metric | Mean | 95% CI | Min | Max |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, group in summary["groups"].items():
        lines.append(
            f"| `{key}` | {group['n']} | `{group['metric']}` | "
            f"{group['mean']:.3f} | {group['ci95']:.3f} | {group['min']:.3f} | {group['max']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(summary: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    groups = summary["groups"]
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), constrained_layout=True)
    fig.suptitle("MetalEnvPool repeated throughput experiments", fontsize=13)

    point_keys = ["point/cpu/zero", "point/mps/zero", "point/cpu/random", "point/mps/random"]
    barplot(
        axes[0],
        [label_for_group(key) for key in point_keys],
        [groups[key]["mean"] / 1e6 for key in point_keys],
        [groups[key]["ci95"] / 1e6 for key in point_keys],
        ylabel="env-steps/s (millions)",
        title="Point dynamics",
        colors=["#4c78a8", "#f58518", "#4c78a8", "#f58518"],
    )

    atari_keys = ["atari_synth/cpu", "atari_synth/mps"]
    barplot(
        axes[1],
        [label_for_group(key) for key in atari_keys],
        [groups[key]["mean"] / 1e3 for key in atari_keys],
        [groups[key]["ci95"] / 1e3 for key in atari_keys],
        ylabel="stacked obs/s (thousands)",
        title="Synthetic Atari preprocessing",
        colors=["#4c78a8", "#f58518"],
    )

    mpe_key = next((key for key in groups if key.startswith("mpe_ppo/")), None)
    if mpe_key:
        barplot(
            axes[2],
            [label_for_group(mpe_key)],
            [groups[mpe_key]["mean"] / 1e6],
            [groups[mpe_key]["ci95"] / 1e6],
            ylabel="env-steps/s (millions)",
            title="MPE PPO loop",
            colors=["#54a24b"],
        )
    else:
        axes[2].axis("off")
    fig.savefig(out_dir / "throughput_summary.pdf")
    fig.savefig(out_dir / "throughput_summary.png", dpi=180)
    plt.close(fig)

    idp = [key for key in groups if key.startswith("sb3_idp/")]
    if idp:
        labels = [label_for_group(key) for key in idp]
        means = [groups[key]["mean"] for key in idp]
        errs = [groups[key]["ci95"] for key in idp]
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        bars = ax.bar(labels, means, yerr=errs, capsize=4, color=["#4c78a8", "#f58518"][: len(labels)])
        ax.set_ylabel("env steps/s")
        ax.set_title("SB3/Gymnasium IDP public-task baselines")
        ax.grid(axis="y", alpha=0.25)
        add_value_labels(ax, bars, fmt="{:.0f}")
        fig.tight_layout()
        fig.savefig(out_dir / "idp_public_baseline.pdf")
        fig.savefig(out_dir / "idp_public_baseline.png", dpi=180)
        plt.close(fig)

    mpe_key = next((key for key in groups if key.startswith("mpe_ppo/")), None)
    if mpe_key:
        group = groups[mpe_key]
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
        fig.suptitle("MPE Simple PPO reward diagnostics", fontsize=13)

        rollout_keys = ["first_mean_rollout_reward", "final_mean_rollout_reward"]
        rollout_bars = axes[0].bar(
            ["first", "final"],
            [group[name]["mean"] for name in rollout_keys],
            yerr=[group[name]["ci95"] for name in rollout_keys],
            capsize=4,
            color=["#4c78a8", "#54a24b"],
        )
        axes[0].set_ylabel("mean rollout reward")
        axes[0].set_title("Training reward")
        axes[0].grid(axis="y", alpha=0.25)
        add_value_labels(axes[0], rollout_bars, fmt="{:.3f}", negative=True)

        eval_bars = axes[1].bar(
            ["eval"],
            [group["eval_mean_episode_return"]["mean"]],
            yerr=[group["eval_mean_episode_return"]["ci95"]],
            capsize=4,
            color=["#72b7b2"],
        )
        axes[1].set_ylabel("episode return")
        axes[1].set_title("Deterministic evaluation")
        axes[1].grid(axis="y", alpha=0.25)
        add_value_labels(axes[1], eval_bars, fmt="{:.2f}", negative=True)
        fig.savefig(out_dir / "mpe_reward_diagnostics.pdf")
        fig.savefig(out_dir / "mpe_reward_diagnostics.png", dpi=180)
        plt.close(fig)


def barplot(
    ax: Any,
    labels: list[str],
    means: list[float],
    errs: list[float],
    *,
    ylabel: str,
    title: str,
    colors: list[str],
) -> None:
    bars = ax.bar(labels, means, yerr=errs, capsize=4, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=22)
    ax.grid(axis="y", alpha=0.25)
    add_value_labels(ax, bars, fmt="{:.1f}")


def add_value_labels(ax: Any, bars: Any, *, fmt: str, negative: bool = False) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = float(bar.get_height())
        if negative and height < 0:
            y = height + 0.06 * span
            va = "bottom"
        else:
            y = height + 0.03 * span
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=8,
        )


def label_for_group(key: str) -> str:
    return (
        key.replace("point/cpu/zero", "CPU zero")
        .replace("point/mps/zero", "MPS zero")
        .replace("point/cpu/random", "CPU random")
        .replace("point/mps/random", "MPS random")
        .replace("atari_synth/cpu", "CPU")
        .replace("atari_synth/mps", "MPS")
        .replace("mpe_ppo/mps", "MPS")
        .replace("sb3_idp/", "IDP ")
    )


if __name__ == "__main__":
    main()
