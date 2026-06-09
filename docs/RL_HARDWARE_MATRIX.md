# RL Hardware Matrix

This matrix uses exact public Gymnasium tasks with the same Stable-Baselines3 PPO runner. It is a hardware/framework control, not a MetalEnvPool tensor-native win claim.

Command:

```bash
uv run --extra video python examples/run_rl_hardware_matrix.py \
  --out-dir artifacts/rl_hardware_matrix/2026-06-09 \
  --run-local \
  --eval-episodes 3

uv run --extra video python examples/run_rl_hardware_matrix.py \
  --out-dir artifacts/rl_hardware_matrix/2026-06-09 \
  --tasks cartpole pendulum idp \
  --run-colab t4 \
  --eval-episodes 3 \
  --reuse-existing

uv run --extra video python examples/run_rl_hardware_matrix.py \
  --out-dir artifacts/rl_hardware_matrix/2026-06-09 \
  --tasks cartpole pendulum idp \
  --run-colab tpu-v5e1 \
  --eval-episodes 3 \
  --reuse-existing
```

Results are generated at `artifacts/rl_hardware_matrix/2026-06-09/hardware_matrix.md`.

| Task | Runner | Device | Timesteps | Steps/s | Eval return | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `CartPole-v1` | Apple M4 SoC CPU backend | `cpu` | 50,000 | 51,015.1 | 359.0 | CPU control |
| `CartPole-v1` | Apple M4 SoC MPS GPU backend | `mps` | 50,000 | 4,512.1 | 376.7 | MLP PPO GPU control |
| `CartPole-v1` | Colab Tesla T4 | `cuda` | 50,000 | 5,351.4 | 432.0 | MLP PPO GPU control |
| `CartPole-v1` | Colab TPU v5e1 runtime | `cpu` | 50,000 | 4,426.4 | 359.0 | TPU runtime; SB3 resolved to CPU fallback |
| `Pendulum-v1` | Apple M4 SoC CPU backend | `cpu` | 50,000 | 53,365.7 | -1,053.3 | CPU control |
| `Pendulum-v1` | Apple M4 SoC MPS GPU backend | `mps` | 50,000 | 6,149.6 | -926.8 | MLP PPO GPU control |
| `Pendulum-v1` | Colab Tesla T4 | `cuda` | 50,000 | 5,899.2 | -997.7 | MLP PPO GPU control |
| `Pendulum-v1` | Colab TPU v5e1 runtime | `cpu` | 50,000 | 4,320.0 | -1,039.2 | TPU runtime; SB3 resolved to CPU fallback |
| `InvertedDoublePendulum-v5` | Apple M4 SoC CPU backend | `cpu` | 50,000 | 17,223.0 | 113.3 | CPU control |
| `InvertedDoublePendulum-v5` | Apple M4 SoC MPS GPU backend | `mps` | 50,000 | 3,077.3 | 131.5 | MLP PPO GPU control |
| `InvertedDoublePendulum-v5` | Colab TPU v5e1 runtime | `cpu` | 50,000 | 1,541.5 | 94.6 | TPU runtime; SB3 resolved to CPU fallback |

Interpretation: the Apple M4 SoC CPU backend wins these small SB3 MLP PPO controls among completed task-matched rows. That is a negative result for generic GPU learner acceleration, and it sharpens the MetalEnvPool thesis: the target is not wrapping SB3 around ordinary Gymnasium loops, but tensor-native environment execution and learner-ready rollout buffers.
