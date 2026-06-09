# Artifacts

Small benchmark and demo artifacts that are referenced by the docs.

- `ppo_rollouts/`: Stable-Baselines3 `InvertedDoublePendulum-v5` model, JSON result, and rollout video.
- `envpool_paper_tasks/2026-06-09/`: exact EnvPool isolated benchmark task names: Atari Pong-v5, MuJoCo Ant-v3, and dm_control cheetah run.
- `envpool_reference/2026-06-09/`: local EnvPool 1.2.5 reference rows for the exact EnvPool paper task set and the conservative `not_won` assessment.
- `breakout_small_cnn/2026-06-09/`: small-CNN Breakout training-loop comparison on M4 plus blocked Colab A100 attempt.
- `mpe_simple_ppo/`: tensor-native `MetalMPESimplePool` PPO run output.
- `report_runs/2026-06-09-main/`: 27-run report batch with raw JSONL, summary JSON/Markdown, and generated figures.
- `rl_hardware_matrix/2026-06-09/`: exact public Gymnasium controls for CartPole, Pendulum, and InvertedDoublePendulum across the Apple M4 SoC CPU backend, Apple M4 SoC MPS GPU backend, Colab CUDA GPU, and Colab TPU-runtime CPU fallback.

These artifacts are benchmark evidence, not training fixtures required by the package.
