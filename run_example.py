"""
Run MORL experiments.
Usage: python run_example.py [cartpole|walker|dst] [variant]
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
            env_name='Walker2d-v5',
            total_timesteps=100000,
        )
        experiment = MultiAlignmentPPOExperiment(config)
        experiment.run()


def run_deep_sea_treasure_experiments():
    
    print("=" * 60)
    print("Running Deep Sea Treasure Experiments")
    print("=" * 60)
    
    for variant in ['A', 'B', 'C']:
        print(f"\n--- Variant {variant} ---")
        config = MultiAlignmentPPOConfig(
            name=f"dst_variant_{variant.lower()}",
            variant=variant,
            env_name='deep-sea-treasure-v0',
            total_timesteps=100000,
        )
        experiment = MultiAlignmentPPOExperiment(config)
        experiment.run()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == 'cartpole':
            run_cartpole_experiments()
        elif cmd == 'walker':
            run_walker_experiments()
        elif cmd == 'dst':
            run_deep_sea_treasure_experiments()
        else:
            print("Usage: python run_example.py [cartpole|walker|dst] [variant]")
    else:
        run_cartpole_experiments()
