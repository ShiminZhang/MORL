"""
Multi-Alignment PPO Experiment
Implements three variants of MORL:
- Variant A: Reward Scalarization
- Variant B: Value/Q-Space Scalarization  
- Variant C: Gradient-Space Combination
"""
import gymnasium as gym
import numpy as np
import torch
import os
import json
from typing import Dict, Any, Optional

from src.experiments.experiment import Experiment, ExperimentConfig
from src.environments import (
    ScalarRewardWrapper,
    SteerableCartPoleWrapper,
    SteerableHumanoidWrapper,
    SteerableWalkerWrapper,
    SteerableDeepSeaTreasureWrapper,
)
from src.trainers import PPOTrainerA, PPOTrainerB, PPOTrainerC

# Environment groups
MUJOCO_ENVS = ('Walker2d-v5', 'Humanoid-v5')
MO_GYM_ENVS = ('deep-sea-treasure-v0',)  # Add future mo_gymnasium envs here


class MultiAlignmentPPOConfig(ExperimentConfig):
    """Configuration for Multi-Alignment PPO experiments."""
    def __init__(
        self,
        name: str,
        variant: str = 'A',  # 'A', 'B', or 'C'
        env_name: str = 'Walker2d-v5',  # 'CartPole-v1' or 'Walker2d-v5'
        learning_rate: float = 3e-4,
        num_steps: int = 128,
        total_timesteps: int = 80000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 4,
        clip_coef: float = 0.2,
        ent_coef: float = 0.001,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.variant = variant
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        is_mujoco = env_name in MUJOCO_ENVS
        is_mo_gym = env_name in MO_GYM_ENVS
        
        if is_mujoco:
            self.num_steps = 2048
            self.total_timesteps = 1000000
            self.update_epochs = 10
            self.ent_coef = 0.0
            self.num_objectives = 3
            self.vf_coef = 0.5 if variant == 'B' else 0.05
        elif is_mo_gym:
            self.num_steps = 64
            self.total_timesteps = 100000
            self.update_epochs = 4
            self.ent_coef = 0.1
            self.vf_coef = 0.5 if variant == 'B' else 0.05
            self.num_objectives = 2
            self.learning_rate = 1e-3
        else:
            # CartPole default
            self.num_objectives = 2


class MultiAlignmentPPOExperiment(Experiment):
    """
    Multi-Alignment PPO Experiment class.
    Supports CartPole and Walker2d environments with three MORL variants.
    """
    
    def __init__(self, config: MultiAlignmentPPOConfig):
        super().__init__(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize environment
        self.env = self._create_environment()

        # Initialize trainer (trainer will explicitly build default model if agent=None)
        self.trainer = self._create_trainer()

        # Expose model from trainer for saving/eval consistency
        self.agent = self.trainer.agent
        
    def _create_environment(self):
        """Create and wrap the environment."""
        env_name = self.config.env_name
        is_mo_gym = env_name in MO_GYM_ENVS
        
        # MO-Gymnasium environments (wrapper creates env internally)
        if is_mo_gym:
            if env_name == 'deep-sea-treasure-v0':
                return SteerableDeepSeaTreasureWrapper()
            raise ValueError(f"Unsupported mo_gym env: {env_name}")
        
        # Standard gymnasium environments
        base_env = gym.make(env_name)
        
        # All variants use the same steerable wrapper (provides vector rewards)
        if env_name == 'CartPole-v1':
            return SteerableCartPoleWrapper(base_env)
        elif env_name == 'Walker2d-v5':
            return SteerableWalkerWrapper(base_env)
        elif env_name == 'Humanoid-v5':
            return SteerableHumanoidWrapper(base_env)
        else:
            raise ValueError(f"Unsupported env_name: {env_name}")
    
    def _create_trainer(self):
        """Create the appropriate PPO trainer based on variant."""
        trainer_kwargs = {
            'agent': None,  # explicit default model is created inside trainer
            'env': self.env,
            'device': self.device,
            'learning_rate': self.config.learning_rate,
            'num_steps': self.config.num_steps,
            'total_timesteps': self.config.total_timesteps,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'update_epochs': self.config.update_epochs,
            'clip_coef': self.config.clip_coef,
            'ent_coef': self.config.ent_coef,
            'vf_coef': self.config.vf_coef,
            'max_grad_norm': self.config.max_grad_norm,
            'batch_size': self.config.batch_size,
            'num_objectives': self.config.num_objectives,
        }
        is_mujoco = self.config.env_name in MUJOCO_ENVS
        
        if self.config.variant == 'A':
            return PPOTrainerA(**trainer_kwargs)
        elif self.config.variant == 'B':
            return PPOTrainerB(**trainer_kwargs)
        else:
            if is_mujoco:
                trainer_kwargs['use_gradient_synthesizer'] = True
            return PPOTrainerC(**trainer_kwargs)
    
    def on_start(self):
        """Called at the start of the experiment."""
        self.logger.info(f"Starting Multi-Alignment PPO Experiment")
        self.logger.info(f"Variant: {self.config.variant}")
        self.logger.info(f"Environment: {self.config.env_name}")
        self.logger.info(f"Config: {self.config.__dict__}")
        self.logger.info(f"Starting experiment: {self.config.variant} on {self.config.env_name}")
    
    def on_end(self):
        """Called at the end of the experiment."""
        # Save model
        model_path = os.path.join(
            self.config.result_dir,
            f"variant_{self.config.variant.lower()}_agent.pth"
        )
        torch.save(self.agent.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Close environment
        self.env.close()
    
    def experiment_main(self):
        """Main experiment execution."""
        # Training
        self.logger.info("Starting training...")
        self.trainer.train()
        
        # Evaluation
        self.logger.info("Starting evaluation...")
        print("\n=== Running Verification ===")
        results = self.trainer.evaluate(test_weights=None)
        
        # Save results
        results_path = os.path.join(
            self.config.result_dir,
            f"variant_{self.config.variant.lower()}_results.json"
        )
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")
    
    def _get_test_weights(self):
        """Get test weights based on number of objectives."""
        if self.config.num_objectives == 2:
            return [
                [1.0, 0.0],  # extreme left
                [0.7, 0.3],  # left
                [0.5, 0.5],  # middle
                [0.3, 0.7],  # right
                [0.0, 1.0]   # extreme right
            ]
        else:  # 3 objectives
            return [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.33, 0.33, 0.34],
                [0.5, 0.3, 0.2],
            ]


def main():
    """Example usage with CartPole."""
    for variant in ['A', 'B', 'C']:
        config = MultiAlignmentPPOConfig(
            name=f"cartpole_variant_{variant.lower()}",
            variant=variant,
            env_name='CartPole-v1',
        )
        MultiAlignmentPPOExperiment(config).run()


def run_deep_sea_treasure(variant: str = 'B'):
    config = MultiAlignmentPPOConfig(
        name=f"dst_variant_{variant.lower()}",
        variant=variant,
        env_name='deep-sea-treasure-v0',
        total_timesteps=50000,
    )
    experiment = MultiAlignmentPPOExperiment(config)
    experiment.run()
    return experiment


if __name__ == "__main__":
    main()
