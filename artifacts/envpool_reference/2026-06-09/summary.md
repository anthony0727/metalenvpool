# Local EnvPool Reference

Exact EnvPool paper task set measured with local EnvPool where available.

| Engine | Task | Num envs | Full env execution | Throughput |
| --- | --- | ---: | --- | ---: |
| envpool | `Pong-v5` | 1 | True | 3,750 steps/s |
| envpool | `Ant-v3` | 1 | True | 6,985 steps/s |
| envpool | `CheetahRun-v1` | 1 | True | 36,405 steps/s |
| envpool | `Pong-v5` | 16 | True | 18,347 steps/s |
| envpool | `Ant-v3` | 16 | True | 37,005 steps/s |
| envpool | `CheetahRun-v1` | 16 | True | 252,518 steps/s |
| envpool | `Pong-v5` | 64 | True | 22,128 steps/s |
| envpool | `Ant-v3` | 64 | True | 47,039 steps/s |
| envpool | `CheetahRun-v1` | 64 | True | 517,870 steps/s |
| metalenvpool | `Pong-v5` | - | False | 36,709 stacked obs/s |
| metalenvpool | `Pong-v5` | - | False | 352,862 stacked obs/s |

Win assessment: **not_won**.

MetalEnvPool has no full-environment execution row for Pong-v5, Ant-v3, or CheetahRun-v1. Partial preprocessing throughput cannot be compared as an EnvPool win.
