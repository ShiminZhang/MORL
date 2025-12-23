# MORL Framework API Documentation

## 1. Experiment Layer

### ExperimentConfig (`src/experiments/experiment.py`)
- Fields: `name`, `data_dir`, `result_dir`, `log_dir`, `K`, `category`
- Auto-creates `./Experiments/{name}/(data|results|logs)`; `experiment_dir = ./Experiments/{name}/`

### Experiment (abstract)
- Lifecycle: `run()` → `on_start()` → `experiment_main()` → `on_end()` (via `end()` optionally)
- Logging: file log (`main@{timestamp}.log`) + Rich console log
- Slurm submission supported
- Abstract methods to implement:
  - `on_start(self)`
  - `experiment_main(self)`
  - `on_end(self)`

### EXAMPLE: MultiAlignmentPPOExperiment (`src/experiments/multi_alignment_ppo.py`)
- Members: `config`, `device`, `env`, `agent`, `trainer`
- Key methods:
  - `_create_environment()`: select wrapper by variant/env
  - `_create_agent()`: CartPole uses discrete agents; Walker2d uses continuous; Variant A uses scalar critic; B/C use vector critic
  - `_create_trainer()`: variant dispatch to `PPOTrainerA/B/C`
  - `_get_test_weights()`: preset eval weights
  - `experiment_main()`: train → eval → save model/results

## 2. Environment Layer

### BaseMORLEnv (`src/environments/base_env.py`)
- Abstract methods:
  - `reset(self, seed: int = None, options: dict = None) -> (obs, info)`
  - `step(self, action) -> (obs, reward, terminated, truncated, info)`
  - `get_reward(self) -> scalar | vector` (returns last step reward; vector preferred)
  - `get_reward_dimension(self) -> int`
- Helpers:
  - `_store_info(info)`
  - `_store_scalar_reward(reward)` (accepts `None`)
  - `_store_vector_reward(reward_vec)` (accepts `None`, stored as list)

### EXAMPLE Implementations
- `ScalarRewardWrapper` (CartPole, Variant A)
  - Obs: original + w(2)
  - `step`: computes **2D vector reward**, stored via `_store_vector_reward`; scalar return is log-only
  - `get_reward_dimension`: 2
- `SteerableCartPoleWrapper` (CartPole, Variant B/C)
  - Obs: original + w(2)
  - `step`: computes 2D vector reward, `_store_vector_reward`
  - `get_reward_dimension`: 2
- `SteerableWalkerWrapper` (Walker2d, 3 objectives)
  - Obs: original + w(3)
  - `step`: computes 3D vector reward stored via `_store_vector_reward`; scalar return is log-only
  - `get_reward_dimension`: 3

## 3. Trainer Layer

### MORLTrainer (abstract, `src/trainers/morl_trainer.py`)
- Abstract: `train(self, train_loader)`, `evaluate(self, eval_loader)`

### PPOTrainerA (Variant A: Reward Scalarization, `src/trainers/ppo_trainer_a.py`)
- For vector rewards + preference weights; scalarization happens **inside trainer**
- Buffers: obs, actions, logprobs, rewards(scalar), values
- Flow: rollout → scalar GAE → PPO update (scalar advantage) → eval prints
- Evaluate returns list of dicts with native Python types: `weight, avg_position, steps`

### PPOTrainerB (Variant B: Value/Q-Space Scalarization, `src/trainers/ppo_trainer_b.py`)
- For vector rewards; scalarize at advantage level
- Extra: `num_objectives`
- Buffers: obs, actions, logprobs, rewards(vector), values(vector), contexts(weights)
- Flow: rollout uses `env.get_reward()`; vector GAE; scalarize advantages (w·A); PPO update (policy uses scalar advantage, value uses vector MSE)

### PPOTrainerC (Variant C: Gradient-Space Combination, `src/trainers/ppo_trainer_c.py`)
- For vector rewards; per-objective PPO loss weighted by context
- Extra: `num_objectives`
- Flow: rollout uses `env.get_reward()`; vector GAE; update computes PPO loss per objective then weighted sum; value loss is vector MSE

## 4. Usage Notes
- Environments expose unified reward access via `get_reward()` / `get_reward_dimension()`
- `reset(seed=None, options=None)`/`step(action)` follow Gymnasium signatures; options can carry preference weights if needed.
- Rich logging enabled in experiments; file logs stored under `Experiments/{name}/logs/`.

