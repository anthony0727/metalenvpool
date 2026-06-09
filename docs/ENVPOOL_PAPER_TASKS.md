# EnvPool Paper Task Matrix

This matrix uses the exact task names from EnvPool's isolated execution benchmark:
Atari Pong-v5, MuJoCo Ant-v3, and dm_control cheetah run. EnvPool's documentation
also notes that older Atari benchmark tables used `PongNoFrameskip-v4` with OpenAI
Baselines wrappers, while current scripts use Gymnasium's `ALE/Pong-v5`.

Sources:

- EnvPool benchmark docs: https://envpool.readthedocs.io/en/stable/content/benchmark.html
- EnvPool current benchmark docs: https://envpool.readthedocs.io/en/latest/content/benchmark.html
- EnvPool dm_control docs: https://envpool.readthedocs.io/en/stable/env/dm_control.html

Command:

```bash
uv run --extra envpool-paper python examples/run_envpool_paper_tasks.py \
  --out-dir artifacts/envpool_paper_tasks/2026-06-09 \
  --physics-steps 200000 \
  --pong-iters 1000
```

Results:

| Paper task | Local id | Status | MetalEnvPool coverage | Throughput | Artifact |
| --- | --- | --- | --- | ---: | --- |
| Atari Pong-v5 | `ALE/Pong-v5` | ok | Atari RGB preprocessing and learner-ready rollout write | MPS 352,862 stacked obs/s; CPU 36,709 | `artifacts/envpool_paper_tasks/2026-06-09/pong.json` |
| MuJoCo Ant-v3 | `Ant-v3` | blocked | not implemented | - | `artifacts/envpool_paper_tasks/2026-06-09/ant.json` |
| dm_control cheetah run | `suite.load("cheetah", "run")` | ok | not implemented; CPU reference only | 52,862 steps/s | `artifacts/envpool_paper_tasks/2026-06-09/cheetah_run.json` |

Interpretation:

MetalEnvPool does not yet reproduce EnvPool's paper benchmark. It only covers
the Atari preprocessing/write part for exact Pong frames. The exact Ant-v3 task
is blocked on this modern Apple stack because Gymnasium's legacy `Ant-v3` path
requires deprecated `mujoco-py`. dm_control `cheetah/run` runs with
`dm-control==1.0.38` and `mujoco==3.6.0`, but this repo currently records it as
a CPU reference because there is no tensor-native Metal cheetah dynamics
implementation yet.
