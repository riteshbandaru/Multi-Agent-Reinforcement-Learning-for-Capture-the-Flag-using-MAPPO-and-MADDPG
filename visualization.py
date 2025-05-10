import matplotlib.pyplot as plt
import os
import csv


def plot_training_curves(log_file):
    episodes, rewards, success_rates = [], [], []
    with open(log_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            success_rates.append(float(row["success_rate"]))

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(log_file), "reward_curve.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, success_rates, label="Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(log_file), "success_rate_curve.png"))
    plt.show()
