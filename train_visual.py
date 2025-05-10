import argparse
import time
import os
import torch
import numpy as np
from ctf_env import TeamEnhancedCaptureTheFlagEnv
from wrappers import MultiAgentCTFWrapper
from logger import Logger
from ppo import PPOAgent

def train_ppo_visual(args):
    # Initialize environment in human render mode for visualization.
    env = TeamEnhancedCaptureTheFlagEnv(render_mode="human")
    env = MultiAgentCTFWrapper(env)

    agent_ids = ["red_0", "red_1", "blue_0", "blue_1"]
    # Create PPO agent for each team member.
    ppo_agents = {}
    for aid in agent_ids:
        obs_shape = env.observation_space.shape  # shape of partial observation e.g., (7, 7, 3)
        ppo_agents[aid] = PPOAgent(obs_shape, action_dim=5, lr=1e-3)

    logger = Logger(log_dir=args.log_dir)
    num_episodes = args.episodes
    max_steps = args.max_steps

    for episode in range(num_episodes):
        # Reset environment and get multi-agent observations.
        obs_dict, _ = env.reset()
        episode_reward = 0
        trajectories = {aid: [] for aid in agent_ids}

        # Display training episode number on the console.
        print(f"Starting Episode {episode}")

        for step in range(max_steps):
            actions = {}
            log_probs = {}
            values = {}
            # For each agent, select an action based on its local observation.
            for aid in agent_ids:
                obs = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)  # (1, C, H, W)
                action, log_prob, value = ppo_agents[aid].select_action(obs)
                actions[aid] = action
                log_probs[aid] = log_prob
                values[aid] = value

            # Step the environment with the action dictionary.
            next_obs_dict, reward, done, _, _ = env.step({aid: actions[aid] for aid in agent_ids})
            episode_reward += reward

            # Store the trajectories for each agent (using common reward).
            for aid in agent_ids:
                traj = {
                    "obs": torch.FloatTensor(obs_dict[aid]).unsqueeze(0),
                    "action": actions[aid],
                    "log_prob": log_probs[aid],
                    "reward": reward,  # team reward is used for every agent
                    "value": values[aid],
                    "done": done,
                }
                trajectories[aid].append(traj)

            obs_dict = next_obs_dict

            # Render the environment visualization.
            env.render()
            # Optional: Slow down visualization for clarity.
            time.sleep(0.1)

            if done:
                break

        # After the episode, update PPO agents on their collected trajectories.
        losses = []
        for aid in agent_ids:
            agent_loss = ppo_agents[aid].update(trajectories[aid])
            losses.append(agent_loss)
        avg_loss = np.mean(losses)

        # Determine success rate (e.g. if a flag was captured).
        success_rate = 1.0 if reward >= 10.0 else 0.0
        logger.log(episode, episode_reward, success_rate)

        print(f"Episode {episode} -- Total Reward: {episode_reward:.2f}, Avg Loss: {avg_loss:.4f}")

    # After training is complete, plot training curves.
    logger.plot()
    # Pause before exit to allow user to view final frame.
    print("Training complete. Close visualization window to exit.")
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes.")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode.")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs and model checkpoints.")
    args = parser.parse_args()

    # Ensure the log directory exists.
    os.makedirs(args.log_dir, exist_ok=True)
    train_ppo_visual(args)


if __name__ == "__main__":
    main()
