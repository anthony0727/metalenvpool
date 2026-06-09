# EnvPool Paper Task Matrix

Exact task names from the EnvPool isolated benchmark task set.

| Paper task | Local id | Status | MetalEnvPool coverage | Throughput | Artifact |
| --- | --- | --- | --- | ---: | --- |
| Atari Pong-v5 | `ALE/Pong-v5` | ok | Atari RGB preprocessing and learner-ready rollout write | MPS 352,862 stacked obs/s; CPU 36,709 | `artifacts/envpool_paper_tasks/2026-06-09/pong.json` |
| MuJoCo Ant-v3 | `Ant-v3` | blocked | not implemented | - | `artifacts/envpool_paper_tasks/2026-06-09/ant.json` |
| dm_control cheetah run | `suite.load('cheetah', 'run')` | ok | not implemented | 52,862 steps/s | `artifacts/envpool_paper_tasks/2026-06-09/cheetah_run.json` |
