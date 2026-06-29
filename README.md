# Decision Transformer trained on a Minari dataset for a MuJoCo environment

An implementation of the Decision Transformer (DT) architecture (Chen et al., 2021), which is an offline Reinforcement Learning algorithm. The checkpoint is trained on Minari datasets and evaluated on corresponding MuJoCo environments.

## Installing dependencies

1. Install `uv` (if not already installed; this might change, please refer to official documentation of `uv`):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install project dependencies:

   ```bash
   uv sync
   ```

## Usage

### 1. Training

To train a checkpoint:

```bash
uv run accelerate launch scripts/train.py
```

### 2. Evaluation

To evaluate a trained checkpoints (or evaluate the checkpoint already provided) and calculate D4RL normalized score:

```bash
uv run python scripts/evaluate.py --env mujoco/halfcheetah/medium-v0 --num-episodes 10
```

#### Evaluation script CLI Arguments

- `--env`: Minari dataset/environment name (e.g. `mujoco/halfcheetah/medium-v0`, `mujoco/walker2d/medium-v0`).
- `--model-path`: Custom checkpoint file path (defaults to `<env_name_with_underscores>_dt.pt`).
- `--target-rtg`: Target return-to-go prompt (defaults to environment baseline, e.g. `6000.0` for HalfCheetah, `5000.0` for Walker2d).
- `--num-episodes`: Number of episodes to run (default: `5`).
- `--render`: Enable human rendering mode to view the MuJoCo UI.

## Benchmarks

Evaluating checkpoints on the medium datasets over 10 episodes yielded the following scores compared to the original paper's baselines:

| Environment | Target RTG | Average Raw Reward | D4RL Normalized Score | Original DT Score |
| :--- | :--- | :--- | :--- | :--- |
| **mujoco/halfcheetah/medium-v0** | 6000 | **5933.65** | **50.05** | 42.6 |
| **mujoco/walker2d/medium-v0** | 5000 | **6064.65** | **132.07** | 74.0 |
