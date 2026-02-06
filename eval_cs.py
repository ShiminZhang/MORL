"""
MORL Evaluation: Cosine Similarity (CS)
Measures alignment between preference weights and achieved return direction.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.environments import make_dm_control_walker
from src.trainers.ppo_trainer_a import VariantA_ContinuousAgent
from src.trainers.ppo_trainer_b import VariantB_ContinuousAgent
from src.trainers.ppo_trainer_c import VariantC_ContinuousAgent
from src.trainers.ppo_trainer_ica import VariantICA_ContinuousAgent

EPS = 1e-8


def cosine_similarity(returns: np.ndarray, prefs: np.ndarray):
    """
    Compute cosine similarity between preference and return direction.

    Args:
        returns: [N, K] return vectors
        prefs: [N, K] preference weights

    Returns:
        mean_cs, cs_per_weight
    """
    returns = np.asarray(returns, dtype=np.float64)
    prefs = np.asarray(prefs, dtype=np.float64)

    n = len(returns)
    cs_per_weight = np.zeros(n)

    for k in range(n):
        w = prefs[k]
        g = returns[k]
        w_norm = np.linalg.norm(w)
        g_norm = np.linalg.norm(g)

        if w_norm < EPS or g_norm < EPS:
            cs_per_weight[k] = 0.0
        else:
            cs_per_weight[k] = np.dot(w, g) / (w_norm * g_norm)

    return float(np.mean(cs_per_weight)), cs_per_weight


def evaluate_model(model, env, weights_list, num_objectives, num_episodes=3, max_steps=1000):
    """Evaluate model with multiple episodes per weight."""
    returns = []
    prefs = []

    for w in weights_list:
        w_array = np.array(w, dtype=np.float32)
        episode_returns = []

        for _ in range(num_episodes):
            obs, _ = env.reset(options={"w": w_array})
            total_rewards = np.zeros(num_objectives)
            steps_taken = 0

            for _ in range(max_steps):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _, _ = model.get_action_and_value(obs_t)
                    action = action.cpu().numpy()[0]

                obs, _, terminated, truncated, _ = env.step(action)
                vec_r = env.get_reward()
                total_rewards += vec_r
                steps_taken += 1

                if terminated or truncated:
                    break

            mean_rewards = total_rewards / steps_taken
            episode_returns.append(mean_rewards)

        avg_return = np.mean(episode_returns, axis=0)
        returns.append(avg_return)
        prefs.append(w_array)

    return np.array(returns), np.array(prefs)


def main():
    weights_list = [
        [1.0, 0.0],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.5, 0.5],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.0, 1.0],
    ]

    models = [
        ("saved_agents/dm_walker_a.pth", "A", VariantA_ContinuousAgent, False),
        ("saved_agents/dm_walker_b.pth", "B", VariantB_ContinuousAgent, True),
        ("saved_agents/dm_walker_c.pth", "C", VariantC_ContinuousAgent, True),
        ("saved_agents/dm_walker_d.pth", "D_ICA", VariantICA_ContinuousAgent, True),
    ]

    env = make_dm_control_walker()
    num_objectives = 2
    all_results = {}

    # Collect returns
    for model_path, variant, AgentClass, needs_num_obj in models:
        print(f"Evaluating {variant}...")
        checkpoint = torch.load(model_path, map_location="cpu")
        model = AgentClass(env, num_objectives) if needs_num_obj else AgentClass(env)
        model.load_state_dict(checkpoint["agent_state_dict"])
        model.eval()

        returns, prefs = evaluate_model(model, env, weights_list, num_objectives, num_episodes=3)
        all_results[variant] = {'returns': returns, 'prefs': prefs}

    env.close()

    # Compute CS for each variant
    all_metrics = {}
    for variant in all_results:
        returns = all_results[variant]['returns']
        prefs = all_results[variant]['prefs']
        mean_cs, cs_per = cosine_similarity(returns, prefs)
        all_metrics[variant] = {'cs': mean_cs, 'cs_per': cs_per}

    # Print results
    print("\n" + "=" * 40)
    print(f"{'Variant':<10} {'Cosine Similarity':<15}")
    print("-" * 40)
    for v in all_metrics:
        print(f"{v:<10} {all_metrics[v]['cs']:<15.4f}")

    # Plot
    variants = list(all_metrics.keys())
    colors = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'D_ICA': 'tab:red'}

    fig, ax = plt.subplots(figsize=(8, 5))
    css = [all_metrics[v]['cs'] for v in variants]
    bars = ax.bar(variants, css, color=[colors[v] for v in variants], edgecolor='k')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Preference Alignment (Higher = Better)')
    ax.set_ylim(0, 1.1)
    for i, cs in enumerate(css):
        ax.text(i, cs + 0.02, f'{cs:.3f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('figures/morl_cs.png', dpi=150)
    print("\nSaved: figures/morl_cs.png")


if __name__ == "__main__":
    main()
