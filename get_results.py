import os
import pandas as pd
import matplotlib.pyplot as plt

def visualize_results(model_number, sumo_case):
    # Define paths
    base_dir = f"./results/{model_number}/{sumo_case}"
    output_dir = f"./results/{model_number}/{sumo_case}"
    
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Load Data ===
    rewards_path = os.path.join(base_dir, "rewards.csv")
    q_values_path = os.path.join(base_dir, "q_values.csv")
    queue_history_path = os.path.join(base_dir, "queue_history.csv")

    if not os.path.exists(rewards_path) or not os.path.exists(q_values_path) or not os.path.exists(queue_history_path):
        print("Error: Some result files are missing.")
        return

    rewards_df = pd.read_csv(rewards_path)
    qvalues_df = pd.read_csv(q_values_path)
    queue_df = pd.read_csv(queue_history_path)

    print(f"Loaded results from {base_dir}")

    # === 2. Plot: Average Q-Values per Episode ===
    avg_q_per_episode = qvalues_df.groupby('episode')[['Q_keep', 'Q_switch']].mean()
    plt.figure(figsize=(10,6))
    plt.plot(avg_q_per_episode.index, avg_q_per_episode['Q_keep'], label="Q_keep (avg)")
    plt.plot(avg_q_per_episode.index, avg_q_per_episode['Q_switch'], label="Q_switch (avg)")
    plt.xlabel('Episode')
    plt.ylabel('Average Q-Value')
    plt.title('Average Q-Values per Episode')
    plt.legend()
    plt.grid(True)
    qvalue_evolution_path = os.path.join(output_dir, "qvalue_avg_per_episode.png")
    plt.savefig(qvalue_evolution_path)
    plt.close()
    print(f"Saved: {qvalue_evolution_path}")

    # === 3. Plot: Queue Length History for Specific Episodes ===
    selected_episodes = [0, 1000, 2000, 3000, 4000, 4999]
    for ep in selected_episodes:
        if ep in queue_df['episode'].unique():
            episode_queue = queue_df[queue_df['episode'] == ep]
            plt.figure(figsize=(10,6))
            plt.plot(episode_queue['step'], episode_queue['queue_left'], label="Queue Left")
            plt.plot(episode_queue['step'], episode_queue['queue_right'], label="Queue Right")
            plt.xlabel('Step')
            plt.ylabel('Queue Length')
            plt.title(f'Queue Length over Time - Episode {ep}')
            plt.legend()
            plt.grid(True)
            queue_plot_path = os.path.join(output_dir, f"queue_episode_{ep}.png")
            plt.savefig(queue_plot_path)
            plt.close()
            print(f"Saved: {queue_plot_path}")
        else:
            print(f"Episode {ep} not found in queue history.")

    # === 4. Plot: Total Reward and Total Delay per Episode ===
    fig, ax1 = plt.subplots(figsize=(12,6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward (Q-Value Sum)', color=color)
    ax1.plot(rewards_df['episode'], rewards_df['sum_q_values'], color=color, label="Total Reward")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Total Sum Queues', color=color)
    ax2.plot(rewards_df['episode'], rewards_df['sum_queues'], color=color, label="Total Sum Queues")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    reward_delay_plot_path = os.path.join(output_dir, "reward_delay_per_episode.png")
    plt.title('Total Reward and Total Delay per Episode')
    plt.savefig(reward_delay_plot_path)
    plt.close()
    print(f"Saved: {reward_delay_plot_path}")

    print(f"✅ All visualizations saved in {output_dir}")

if __name__ == "__main__":
    # Example call
    visualize_results(model_number=1, sumo_case="hl_5000_q")
