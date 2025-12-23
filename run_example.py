"""
Example script to run MORL experiments.
"""
import sys
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.experiments.multi_alignment_ppo import MultiAlignmentPPOExperiment, MultiAlignmentPPOConfig


def run_cartpole_experiments():
    """Run all three variants on CartPole."""
    print("=" * 60)
    print("Running CartPole Experiments")
    print("=" * 60)
    
    for variant in ['A', 'B', 'C']:
        print(f"\n--- Variant {variant} ---")
        config = MultiAlignmentPPOConfig(
            name=f"cartpole_variant_{variant.lower()}",
            variant=variant,
            env_name='CartPole-v1',
            total_timesteps=80000,  # Reduced for quick testing
        )
        experiment = MultiAlignmentPPOExperiment(config)
        experiment.run()


def run_walker_experiments():
    """Run all three variants on Walker2d."""
    print("=" * 60)
    print("Running Walker2d Experiments")
    print("=" * 60)
    
    for variant in ['A', 'B', 'C']:
        print(f"\n--- Variant {variant} ---")
        config = MultiAlignmentPPOConfig(
            name=f"walker_variant_{variant.lower()}",
            variant=variant,
            env_name='Walker2d-v4',
            total_timesteps=100000,  # Reduced for quick testing
        )
        experiment = MultiAlignmentPPOExperiment(config)
        experiment.run()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cartpole':
            run_cartpole_experiments()
        elif sys.argv[1] == 'walker':
            run_walker_experiments()
        else:
            print("Usage: python run_example.py [cartpole|walker]")
    else:
        # Run CartPole by default
        run_cartpole_experiments()

