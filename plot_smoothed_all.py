"""
Smoothed evaluation plots for all DM Control environments.
"""
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.environments import make_dm_control_walker, make_dm_control_hopper, make_dm_control_humanoid
from src.trainers.ppo_trainer_a import VariantA_ContinuousAgent
from src.trainers.ppo_trainer_b import VariantB_ContinuousAgent
from src.trainers.ppo_trainer_c import VariantC_ContinuousAgent
from src.trainers.ppo_trainer_ica import VariantICA_ContinuousAgent


def moving_average(data, window=50):
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def evaluate_model(model, env, weights_list, max_steps=1000):
    results = {}

    for w in weights_list:
        w_array = np.array(w, dtype=np.float32)
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

        results[tuple(w)] = step_rewards

    return results


def plot_env(env_name, make_fn, model_prefix, save_name):
    models = [
        (f"saved_agents/{model_prefix}_a.pth", "A", VariantA_ContinuousAgent, False),
        (f"saved_agents/{model_prefix}_b.pth", "B", VariantB_ContinuousAgent, True),
        (f"saved_agents/{model_prefix}_c.pth", "C", VariantC_ContinuousAgent, True),
        (f"saved_agents/{model_prefix}_d.pth", "D_ICA", VariantICA_ContinuousAgent, True),
    ]

    weights_list = [
        [1.0, 0.0],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.0, 1.0],
    ]

    env = make_fn()
    num_objectives = 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(weights_list)))

    for idx, (model_path, variant, AgentClass, needs_num_obj) in enumerate(models):
        ax = axes[idx]

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model = AgentClass(env, num_objectives) if needs_num_obj else AgentClass(env)
            model.load_state_dict(checkpoint["agent_state_dict"])
            model.eval()

            results = evaluate_model(model, env, weights_list)

            for (w, rewards), color in zip(results.items(), colors):
                smoothed = moving_average(np.array(rewards), window=50)
                label = f"w={[round(x, 2) for x in w]}"
                ax.plot(smoothed, color=color, label=label, linewidth=2)

            ax.set_title(f'Variant {variant}', fontsize=12)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Weighted Reward (w·r)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.1)

        except Exception as e:
            ax.set_title(f'Variant {variant} - Error')
            ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes)

    env.close()

    plt.suptitle(f'{env_name} - Smoothed Evaluation Curves', fontsize=14)
    plt.tight_layout()
    # Auto-increment filename: save_name_01, _02, ...
    existing = sorted(glob.glob(f'figures/{save_name}_[0-9][0-9].png'))
    if existing:
        last_num = int(os.path.basename(existing[-1]).replace(f'{save_name}_', '').replace('.png', ''))
        next_num = last_num + 1
    else:
        next_num = 1
    out_path = f'figures/{save_name}_{next_num:02d}.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    print("Plotting DM Control Walker...")
    plot_env("DM Control Walker", make_dm_control_walker, "dm_walker", "smoothed_dm_walker")

    print("Plotting DM Control Hopper...")
    plot_env("DM Control Hopper", make_dm_control_hopper, "dm_hopper", "smoothed_dm_hopper")

    print("Plotting DM Control Humanoid...")
    plot_env("DM Control Humanoid", make_dm_control_humanoid, "dm_humanoid", "smoothed_dm_humanoid")


if __name__ == "__main__":
    main()
