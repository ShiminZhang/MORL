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
from typing import Dict, Any, Optional

from src.experiments.experiment import Experiment, ExperimentConfig
from src.environments import ScalarRewardWrapper, SteerableCartPoleWrapper, SteerableWalkerWrapper
from src.agents import Agent, VectorAgent, ContinuousScalarAgent, ContinuousVectorAgent
from src.trainers import PPOTrainerA, PPOTrainerB, PPOTrainerC


class MultiAlignmentPPOConfig(ExperimentConfig):
    """Configuration for Multi-Alignment PPO experiments."""
    def __init__(
        self,
        name: str,
        variant: str = 'A',  # 'A', 'B', or 'C'
        env_name: str = 'CartPole-v1',  # 'CartPole-v1' or 'Walker2d-v4'
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
        
        # Adjust for continuous environments
        if env_name == 'Walker2d-v4':
            self.num_steps = 2048
            self.total_timesteps = 1000000
            self.update_epochs = 10
            self.ent_coef = 0.0
            self.num_objectives = 3
        else:
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
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Initialize trainer
        self.trainer = self._create_trainer()
        
    def _create_environment(self):
        """Create and wrap the environment."""
        base_env = gym.make(self.config.env_name)
        
        if self.config.variant == 'A':
            if self.config.env_name == 'CartPole-v1':
                return ScalarRewardWrapper(base_env)
            else:
                # For Walker2d, Variant A uses scalar reward wrapper
                return SteerableWalkerWrapper(base_env)
        else:
            if self.config.env_name == 'CartPole-v1':
                return SteerableCartPoleWrapper(base_env)
            else:
                return SteerableWalkerWrapper(base_env)
    
    def _create_agent(self):
        """Create the appropriate agent based on variant and environment."""
        is_continuous = self.config.env_name == 'Walker2d-v4'
        
        if self.config.variant == 'A':
            if is_continuous:
                agent = ContinuousScalarAgent(self.env)
            else:
                agent = Agent(self.env)
        else:
            if is_continuous:
                agent = ContinuousVectorAgent(self.env, num_objectives=self.config.num_objectives)
            else:
                agent = VectorAgent(self.env)
        
        return agent.to(self.device)
    
    def _create_trainer(self):
        """Create the appropriate PPO trainer based on variant."""
        trainer_kwargs = {
            'agent': self.agent,
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
        }
        
        if self.config.variant == 'A':
            trainer_kwargs['num_objectives'] = self.config.num_objectives
            return PPOTrainerA(**trainer_kwargs)
        elif self.config.variant == 'B':
            trainer_kwargs['num_objectives'] = self.config.num_objectives
            return PPOTrainerB(**trainer_kwargs)
        else:  # Variant C
            trainer_kwargs['num_objectives'] = self.config.num_objectives
            return PPOTrainerC(**trainer_kwargs)
    
    def on_start(self):
        """Called at the start of the experiment."""
        self.logger.info(f"Starting Multi-Alignment PPO Experiment")
        self.logger.info(f"Variant: {self.config.variant}")
        self.logger.info(f"Environment: {self.config.env_name}")
        self.logger.info(f"Config: {self.config.__dict__}")
    
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
        
        test_weights = self._get_test_weights()
        results = self.trainer.evaluate(test_weights=test_weights)
        
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
    """Example usage."""
    # Example: Variant A with CartPole
    config_a = MultiAlignmentPPOConfig(
        name="cartpole_variant_a",
        variant='A',
        env_name='CartPole-v1',
    )
    experiment_a = MultiAlignmentPPOExperiment(config_a)
    experiment_a.run()
    
    # Example: Variant B with CartPole
    config_b = MultiAlignmentPPOConfig(
        name="cartpole_variant_b",
        variant='B',
        env_name='CartPole-v1',
    )
    experiment_b = MultiAlignmentPPOExperiment(config_b)
    experiment_b.run()
    
    # Example: Variant C with CartPole
    config_c = MultiAlignmentPPOConfig(
        name="cartpole_variant_c",
        variant='C',
        env_name='CartPole-v1',
    )
    experiment_c = MultiAlignmentPPOExperiment(config_c)
    experiment_c.run()


if __name__ == "__main__":
    main()

