# Changelog

## Refactoring: Absolute Imports and Separate Trainers

### Changes Made

1. **Absolute Imports**: All relative imports (`.module`) have been replaced with absolute imports (`src.module`)
   - All `__init__.py` files now use absolute imports
   - All trainer files use absolute imports
   - All experiment files use absolute imports

2. **Separate Trainers**: Split the unified `PPOTrainer` into three independent trainers:
   - `PPOTrainerA`: Variant A - Reward Scalarization
   - `PPOTrainerB`: Variant B - Value/Q-Space Scalarization
   - `PPOTrainerC`: Variant C - Gradient-Space Combination
   - Removed the old `ppo_trainer.py` file

3. **Python Path Setup**: 
   - Added path setup in `src/__init__.py` to automatically add project root to Python path
   - Updated `src/morl.py` and `run_example.py` to include path setup
   - Created `setup.py` for package installation

### File Structure

```
src/
├── trainers/
│   ├── ppo_trainer_a.py    # Variant A trainer
│   ├── ppo_trainer_b.py    # Variant B trainer
│   ├── ppo_trainer_c.py    # Variant C trainer
│   ├── morl_trainer.py     # Base trainer class
│   └── __init__.py         # Exports all trainers
```

### Usage

All imports now use absolute paths:
```python
from src.trainers import PPOTrainerA, PPOTrainerB, PPOTrainerC
from src.environments import ScalarRewardWrapper
from src.agents import Agent, VectorAgent
```

The project root is automatically added to Python path when importing `src`, so absolute imports work without additional setup.

