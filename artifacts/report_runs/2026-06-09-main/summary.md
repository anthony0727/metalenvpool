# Report Experiment Summary

- started_at: `2026-06-09T17:05:03+0900`
- seeds: `[7, 11, 13]`
- runs: `27`

| Group | n | Metric | Mean | 95% CI | Min | Max |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `atari_synth/cpu` | 3 | `stacked_observations_per_second` | 31116.473 | 6331.519 | 27689.108 | 37573.117 |
| `atari_synth/mps` | 3 | `stacked_observations_per_second` | 334378.592 | 60718.921 | 275429.735 | 380372.015 |
| `mpe_ppo/mps` | 3 | `env_steps_per_second` | 1215692.958 | 196587.826 | 1018724.847 | 1347084.295 |
| `point/cpu/random` | 3 | `env_steps_per_second` | 50341314.591 | 3148469.475 | 47354443.098 | 52859516.368 |
| `point/cpu/zero` | 3 | `env_steps_per_second` | 63262475.075 | 158455.337 | 63106222.848 | 63376607.574 |
| `point/mps/random` | 3 | `env_steps_per_second` | 143032202.806 | 34948659.850 | 119442881.775 | 177989034.341 |
| `point/mps/zero` | 3 | `env_steps_per_second` | 206118987.372 | 53499480.478 | 152407623.715 | 241428965.661 |
| `sb3_idp/cpu` | 3 | `env_steps_per_second` | 17402.211 | 166.748 | 17270.019 | 17561.083 |
| `sb3_idp/mps` | 3 | `env_steps_per_second` | 3075.896 | 60.680 | 3014.036 | 3109.155 |
