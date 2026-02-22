"""
Evaluate saved models and create smoothed plots.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.environments import make_dm_control_walker
from src.trainers.ppo_trainer_a import VariantA_ContinuousAgent
from src.trainers.ppo_trainer_b import VariantB_ContinuousAgent
from src.trainers.ppo_trainer_c import VariantC_ContinuousAgent
from src.trainers.ppo_trainer_ica import VariantICA_ContinuousAgent


def moving_average(data, window=50):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def evaluate_model(model, env, weights_list, episodes=1, max_steps=1000):
    """Evaluate model with different weight settings."""
    results = {}

    for w in weights_list:
        w_array = np.array(w, dtype=np.float32)
        episode_rewards = []

        for _ in range(episodes):
            obs, _ = env.reset(options={"w": w_array})
            step_rewards = []

            for _ in range(max_steps):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _, _ = model.get_action_and_value(obs_t)
                    action = action.cpu().numpy()[0]

                obs, _, terminated, truncated, _ = env.step(action)
                vec_r = env.get_reward()
                weighted_r = float(np.dot(w_array, vec_r))
                step_rewards.append(weighted_r)

                if terminated or truncated:
                    break

            episode_rewards.append(step_rewards)

        results[tuple(w)] = episode_rewards

    return results


def plot_smoothed(results, title, save_path, window=50):
    """Plot smoothed evaluation curves."""
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (w, episodes), color in zip(results.items(), colors):
        # Average across episodes, then smooth
        all_rewards = np.array(episodes)
        mean_rewards = all_rewards.mean(axis=0)
        smoothed = moving_average(mean_rewards, window)

        label = f"w={[round(x, 2) for x in w]}"
        plt.plot(smoothed, color=color, label=label, linewidth=2)

    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Weighted Reward (w·r)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # Weight settings to evaluate
    weights_list = [
        [1.0, 0.0],    # Pure velocity
        [0.75, 0.25],
        [0.5, 0.5],    # Balanced
        [0.25, 0.75],
        [0.0, 1.0],    # Pure energy
    ]

    # Model paths, names, agent classes, and whether they need num_objectives
    models = [
        ("saved_agents/dm_walker_a.pth", "A", VariantA_ContinuousAgent, False),
        ("saved_agents/dm_walker_b.pth", "B", VariantB_ContinuousAgent, True),
        ("saved_agents/dm_walker_c.pth", "C", VariantC_ContinuousAgent, True),
        ("saved_agents/dm_walker_d.pth", "D_ICA", VariantICA_ContinuousAgent, True),
    ]

    env = make_dm_control_walker()
    num_objectives = 2

    Path("figures/smoothed").mkdir(parents=True, exist_ok=True)

    for model_path, variant, AgentClass, needs_num_obj in models:
        print(f"\nEvaluating Variant {variant}...")

        checkpoint = torch.load(model_path, map_location="cpu")
        if needs_num_obj:
            model = AgentClass(env, num_objectives)
        else:
            model = AgentClass(env)
        model.load_state_dict(checkpoint["agent_state_dict"])
        model.eval()

        results = evaluate_model(model, env, weights_list, episodes=1, max_steps=1000)

        save_path = f"figures/smoothed/smoothed_{variant}_dm_walker.png"
        plot_smoothed(
            results,
            f"Variant {variant} - Smoothed Eval (dm_control_walker)",
            save_path,
            window=50
        )

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
