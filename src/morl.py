"""
Main entry point for MORL experiments.
"""
import sys
import argparse
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.experiments.multi_alignment_ppo import MultiAlignmentPPOExperiment, MultiAlignmentPPOConfig


def main():
    parser = argparse.ArgumentParser(description='MORL Experiments')
    parser.add_argument('--variant', type=str, default='A', choices=['A', 'B', 'C'],
                        help='MORL variant: A (Reward Scalarization), B (Value Scalarization), C (Gradient Mixing)')
    parser.add_argument('--env', type=str, default='Walker2d-v5', 
                        choices=['CartPole-v1', 'Walker2d-v5', 'Humanoid-v5'],
                        help='Environment name')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--total_timesteps', type=int, default=None,
                        help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Set default experiment name
    if args.name is None:
        args.name = f"{args.env.lower()}_variant_{args.variant.lower()}"
    
    # Create config
    config_kwargs = {
        'name': args.name,
        'variant': args.variant,
        'env_name': args.env,
        'learning_rate': args.learning_rate,
    }
    
    if args.total_timesteps is not None:
        config_kwargs['total_timesteps'] = args.total_timesteps
    
    config = MultiAlignmentPPOConfig(**config_kwargs)
    
    # Create and run experiment
    experiment = MultiAlignmentPPOExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
