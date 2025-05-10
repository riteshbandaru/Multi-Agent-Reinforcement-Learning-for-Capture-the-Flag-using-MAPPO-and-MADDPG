import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def load_log(log_file):
    return pd.read_csv(log_file)

def plot_reward_comparison(df1, df2, algo1_name="PPO", algo2_name="MADDPG"):
    plt.figure(figsize=(10, 6))
    plt.plot(df1['episode'], df1['reward'], label=f"{algo1_name} Reward")
    plt.plot(df2['episode'], df2['reward'], label=f"{algo2_name} Reward")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_comparison.png")
    plt.show()

def plot_success_rate_comparison(df1, df2, algo1_name="PPO", algo2_name="MADDPG"):
    plt.figure(figsize=(10, 6))
    plt.plot(df1['episode'], df1['success_rate'], label=f"{algo1_name} Success Rate")
    plt.plot(df2['episode'], df2['success_rate'], label=f"{algo2_name} Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("success_rate_comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log1", type=str, required=True, help="Path to log CSV file for algorithm 1 (e.g. PPO)")
    parser.add_argument("--log2", type=str, required=True, help="Path to log CSV file for algorithm 2 (e.g. MADDPG)")
    parser.add_argument("--algo1", type=str, default="PPO", help="Name for algorithm 1")
    parser.add_argument("--algo2", type=str, default="MADDPG", help="Name for algorithm 2")
    args = parser.parse_args()

    if not os.path.exists(args.log1) or not os.path.exists(args.log2):
        print("One or both log files were not found.")
        return

    df1 = load_log(args.log1)
    df2 = load_log(args.log2)

    plot_reward_comparison(df1, df2, args.algo1, args.algo2)
    plot_success_rate_comparison(df1, df2, args.algo1, args.algo2)

if __name__ == "__main__":
    main()
