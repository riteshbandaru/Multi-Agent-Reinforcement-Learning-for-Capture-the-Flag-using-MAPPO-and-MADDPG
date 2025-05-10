import os
import csv
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, log_dir="logs", filename="training_log.csv"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_path = os.path.join(self.log_dir, filename)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "success_rate"])

    def log(self, episode, reward, success_rate):
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, success_rate])

    def plot(self):
        episodes, rewards, success_rates = [], [], []
        with open(self.file_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["reward"]))
                success_rates.append(float(row["success_rate"]))
        plt.figure()
        plt.plot(episodes, rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "reward_plot.png"))

        plt.figure()
        plt.plot(episodes, success_rates, label="Success Rate")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "success_rate_plot.png"))
        plt.show()
