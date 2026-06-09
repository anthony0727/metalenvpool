# Breakout Small-CNN Training Loop

This is the current small deep-RL benchmark surface. It compares a tiny CNN
actor-critic loop on Breakout with two env frontends:

- EnvPool `Breakout-v5`: full Atari env execution, returns stacked `4x84x84`
  observations.
- MetalEnvPool path: Gymnasium/ALE `ALE/Breakout-v5` full env execution, plus
  MetalEnvPool fused Metal preprocessing/frame-stack/rollout write on MPS.

Command:

```bash
uv run --extra envpool-compare python examples/train_breakout_small_cnn_compare.py \
  --backend envpool \
  --device auto \
  --num-envs 16 \
  --rollout-steps 32 \
  --total-timesteps 32768 \
  --output-json artifacts/breakout_small_cnn/2026-06-09/envpool_m4_16x32.json

uv run --extra envpool-paper python examples/train_breakout_small_cnn_compare.py \
  --backend metalenvpool \
  --device auto \
  --num-envs 16 \
  --rollout-steps 32 \
  --total-timesteps 32768 \
  --output-json artifacts/breakout_small_cnn/2026-06-09/metalenvpool_m4_16x32.json
```

Results:

| Backend | Env frontend | Learner device | Timesteps | Throughput | Full env execution |
| --- | --- | --- | ---: | ---: | --- |
| EnvPool | `Breakout-v5` | MPS | 32,768 | 3,058 steps/s | yes |
| MetalEnvPool path | `ALE/Breakout-v5` via Gymnasium/ALE + Metal preprocessing | MPS | 32,768 | 1,502 steps/s | yes |

A Colab A100 run was attempted with:

```bash
colab run --gpu A100 examples/train_breakout_small_cnn_compare.py \
  --backend envpool \
  --device auto \
  --num-envs 16 \
  --rollout-steps 32 \
  --total-timesteps 32768 \
  --output-json breakout_envpool_a100.json
```

The Colab backend rejected A100 for this account/quota, so there is no valid
A100 row yet.

Interpretation:

Current status is **not_won**. Breakout is trainable on M4 with a small CNN, but
the current MetalEnvPool path still pays Python/Gymnasium ALE stepping overhead.
The Metal preprocessing kernel is fast, but full Breakout throughput is slower
than EnvPool's full env path on the same M4 machine. A valid win needs a native
or lower-overhead Atari env frontend, not only Metal preprocessing.
