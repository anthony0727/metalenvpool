# RL Hardware Matrix

Exact public Gymnasium tasks using the same Stable-Baselines3 PPO runner.
These rows compare framework/hardware behavior, not a MetalEnvPool tensor-native task win.

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
