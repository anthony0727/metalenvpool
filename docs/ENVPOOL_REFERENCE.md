# Local EnvPool Reference

This benchmark runs local EnvPool 1.2.5 on the exact EnvPool paper task set and
keeps MetalEnvPool rows separate unless they include full environment execution.

Prerequisite on this Mac:

```bash
brew install qt@5
```

EnvPool's macOS wheel imports procgen bindings that link against Qt 5. Without
that Homebrew framework, importing `envpool` failed before any benchmark ran.

Command:

```bash
uv run --extra envpool-compare python examples/run_envpool_reference.py \
  --out-dir artifacts/envpool_reference/2026-06-09 \
  --num-envs 1 16 64 \
  --steps-small 5000 \
  --steps-large 2000
```

Results:

| Engine | Task | Num envs | Full env execution | Throughput |
| --- | --- | ---: | --- | ---: |
| envpool | `Pong-v5` | 1 | true | 3,750 steps/s |
| envpool | `Ant-v3` | 1 | true | 6,985 steps/s |
| envpool | `CheetahRun-v1` | 1 | true | 36,405 steps/s |
| envpool | `Pong-v5` | 16 | true | 18,347 steps/s |
| envpool | `Ant-v3` | 16 | true | 37,005 steps/s |
| envpool | `CheetahRun-v1` | 16 | true | 252,518 steps/s |
| envpool | `Pong-v5` | 64 | true | 22,128 steps/s |
| envpool | `Ant-v3` | 64 | true | 47,039 steps/s |
| envpool | `CheetahRun-v1` | 64 | true | 517,870 steps/s |
| metalenvpool | `Pong-v5` | - | false | 352,862 stacked obs/s |

Win assessment: **not_won**.

MetalEnvPool has no full-environment execution row for `Pong-v5`, `Ant-v3`, or
`CheetahRun-v1`. The Pong Metal row is only Atari RGB preprocessing and
learner-ready rollout write; it cannot be compared as an EnvPool task win.
