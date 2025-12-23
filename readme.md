# MORL: Multi-Objective Reinforcement Learning Framework

A flexible framework for implementing and experimenting with Multi-Objective Reinforcement Learning (MORL) algorithms.

## Features

- **Three MORL Variants**:
  - **Variant A**: Reward Scalarization
  - **Variant B**: Value/Q-Space Scalarization
  - **Variant C**: Gradient-Space Combination

- **Multiple Environments**:
  - CartPole-v1 (discrete actions, 2 objectives)
  - Walker2d-v4 (continuous actions, 3 objectives)

- **Modular Architecture**:
  - Environment wrappers
  - Agent implementations
  - PPO trainer
  - Experiment framework

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

This will add the project root to Python path, allowing absolute imports from `src.*`.

## Usage

### Command Line

```bash
# Run Variant A with CartPole
python -m src.morl --variant A --env CartPole-v1 --name my_experiment

# Run Variant B with Walker2d
python -m src.morl --variant B --env Walker2d-v4 --name walker_experiment

# Run Variant C with custom timesteps
python -m src.morl --variant C --env CartPole-v1 --total_timesteps 100000
```

### Python API

```python
from src.experiments.multi_alignment_ppo import MultiAlignmentPPOExperiment, MultiAlignmentPPOConfig

# Create config
config = MultiAlignmentPPOConfig(
    name="my_experiment",
    variant='A',
    env_name='CartPole-v1',
    total_timesteps=80000,
)

# Run experiment
experiment = MultiAlignmentPPOExperiment(config)
experiment.run()
```

## Project Structure

```
MORL/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── discrete_agent.py
│   │   └── continuous_agent.py
│   ├── environments/     # Environment wrappers
│   │   ├── cartpole_wrappers.py
│   │   └── walker_wrapper.py
│   ├── trainers/         # Training algorithms
│   │   ├── morl_trainer.py
│   │   └── ppo_trainer.py
│   ├── experiments/      # Experiment classes
│   │   ├── experiment.py
│   │   └── multi_alignment_ppo.py
│   └── utils/            # Utility functions
│       └── paths.py
├── data/                 # Data directory
├── models/               # Saved models
├── Experiments/          # Experiment outputs
└── requirements.txt
```

## Experiment Outputs

Experiments save:
- Model weights: `Experiments/{name}/results/variant_{variant}_agent.pth`
- Results: `Experiments/{name}/results/variant_{variant}_results.json`
- Logs: `Experiments/{name}/logs/main@{timestamp}.log`

## License

MIT

