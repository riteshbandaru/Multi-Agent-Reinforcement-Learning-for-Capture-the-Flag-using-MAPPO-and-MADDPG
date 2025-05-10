import argparse
import os
import numpy as np
import torch
from ctf_env import TeamEnhancedCaptureTheFlagEnv
from wrappers import MultiAgentCTFWrapper
from logger import Logger
from ppo import PPOAgent
from maddpg import MADDPG
from mappo import MAPPOAgent   # New import for MAPPO

"""
This training script allows you to choose the RL algorithm (PPO, MADDPG, or MAPPO) to train the agents.
"""

def train_ppo(args):
    env = TeamEnhancedCaptureTheFlagEnv(render_mode=None)
    env = MultiAgentCTFWrapper(env)
    agent_ids = ["red_0", "red_1", "blue_0", "blue_1"]
    ppo_agents = {}
    for agent_id in agent_ids:
        obs_shape = env.observation_space.shape
        ppo_agents[agent_id] = PPOAgent(obs_shape, action_dim=5, lr=1e-3)
    logger = Logger(log_dir=args.log_dir)
    num_episodes = args.episodes
    max_steps = args.max_steps
    for episode in range(num_episodes):
        obs_dict, _ = env.reset()
        episode_reward = 0
        trajectories = {aid: [] for aid in agent_ids}
        for step in range(max_steps):
            actions = {}
            log_probs = {}
            values = {}
            for aid in agent_ids:
                obs = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)
                action, log_prob, value = ppo_agents[aid].select_action(obs)
                actions[aid] = action
                log_probs[aid] = log_prob
                values[aid] = value
            next_obs_dict, reward, done, _, _ = env.step({aid: actions[aid] for aid in agent_ids})
            episode_reward += reward
            for aid in agent_ids:
                traj = {
                    "obs": torch.FloatTensor(obs_dict[aid]).unsqueeze(0),
                    "action": actions[aid],
                    "log_prob": log_probs[aid],
                    "reward": reward,
                    "value": values[aid],
                    "done": done,
                }
                trajectories[aid].append(traj)
            obs_dict = next_obs_dict
            if done:
                break
        losses = []
        for aid in agent_ids:
            agent_loss = ppo_agents[aid].update(trajectories[aid])
            losses.append(agent_loss)
        avg_loss = np.mean(losses)
        success_rate = 1.0 if reward >= 10.0 else 0.0
        logger.log(episode, episode_reward, success_rate)
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Loss: {avg_loss:.4f}")
        if episode % 50 == 0:
            ckpt = {aid: ppo_agents[aid].policy.state_dict() for aid in agent_ids}
            torch.save(ckpt, os.path.join(args.log_dir, f"ppo_checkpoint_ep{episode}.pth"))
    logger.plot()


def train_maddpg(args):
    env = TeamEnhancedCaptureTheFlagEnv(render_mode=None)
    full_obs, _ = env.reset()
    full_obs_dim = np.prod(full_obs.shape)  # e.g., 15*15*3 = 675
    obs_dim = (2 * env.vision_range + 1) ** 2 * 3  # e.g., 7*7*3 = 147
    action_dim = 5
    # Enable communication with CommNet.
    maddpg = MADDPG(obs_dim, action_dim, full_obs_dim, action_dim * 4, num_agents=4, use_commnet=True)
    logger = Logger(log_dir=args.log_dir)
    num_episodes = args.episodes
    max_steps = args.max_steps
    for episode in range(num_episodes):
        full_obs, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            local_obs_dict = env.get_agent_observations()
            local_obs_batch = [local_obs_dict[aid].flatten() for aid in ["red_0", "red_1", "blue_0", "blue_1"]]
            full_obs = env.get_observation().flatten()
            actions = maddpg.act(local_obs_batch, noise=0.05)
            next_full_obs, reward, done, _, _ = env.step(actions)
            episode_reward += reward
            next_local_obs_dict = env.get_agent_observations()
            next_local_obs_batch = [next_local_obs_dict[aid].flatten() for aid in ["red_0", "red_1", "blue_0", "blue_1"]]
            next_full_obs_flat = env.get_observation().flatten()
            joint_action = []
            for a in actions:
                one_hot = np.zeros(action_dim)
                one_hot[a] = 1
                joint_action.extend(one_hot)
            joint_action = np.array(joint_action)
            maddpg.push(local_obs_batch, full_obs, joint_action, reward, next_local_obs_batch, next_full_obs_flat, done)
            if done:
                break
        loss = maddpg.update(batch_size=256)
        success_rate = 1.0 if reward >= 10.0 else 0.0
        logger.log(episode, episode_reward, success_rate)
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}")
        if episode % 50 == 0:
            ckpt = {"agent_" + str(i): maddpg.agents[i].actor.state_dict() for i in range(4)}
            torch.save(ckpt, os.path.join(args.log_dir, f"maddpg_checkpoint_ep{episode}.pth"))
    logger.plot()

def train_mappo(args):
    """
    Training loop for MAPPO.
    MAPPO is on-policy, so we collect trajectories from the current policy
    and update with a PPO-style clipped objective.
    """
    env = TeamEnhancedCaptureTheFlagEnv(render_mode=None)
    env = MultiAgentCTFWrapper(env)
    agent_ids = ["red_0", "red_1", "blue_0", "blue_1"]
    mappo_agents = {}
    # Use the wrapper's observation space for local (partial) observations.
    obs_shape = env.observation_space.shape
    for agent_id in agent_ids:
        mappo_agents[agent_id] = MAPPOAgent(obs_shape, action_dim=5, lr=3e-4)
    logger = Logger(log_dir=args.log_dir)
    num_episodes = args.episodes
    max_steps = args.max_steps
    for episode in range(num_episodes):
        obs_dict, _ = env.reset()
        episode_reward = 0
        # For MAPPO, collect trajectories for each agent in the current policy.
        trajectories = {aid: [] for aid in agent_ids}
        for step in range(max_steps):
            actions = {}
            for aid in agent_ids:
                obs = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)  # Shape: (1, C, H, W)
                action, log_prob, value = mappo_agents[aid].select_action(obs)
                actions[aid] = action
                traj = {
                    "obs": obs,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                }
                trajectories[aid].append(traj)
            next_obs_dict, reward, done, _, _ = env.step({aid: actions[aid] for aid in agent_ids})
            episode_reward += reward
            for aid in agent_ids:
                trajectories[aid][-1]["reward"] = reward
                trajectories[aid][-1]["done"] = done
            obs_dict = next_obs_dict
            if done:
                break
        losses = []
        for aid in agent_ids:
            agent_loss = mappo_agents[aid].update(trajectories[aid])
            losses.append(agent_loss)
        avg_loss = np.mean(losses)
        success_rate = 1.0 if reward >= 10.0 else 0.0
        logger.log(episode, episode_reward, success_rate)
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Loss: {avg_loss:.4f}")
        if episode % 50 == 0:
            ckpt = {aid: mappo_agents[aid].policy.state_dict() for aid in agent_ids}
            torch.save(ckpt, os.path.join(args.log_dir, f"mappo_checkpoint_ep{episode}.pth"))
    logger.plot()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "maddpg", "mappo"],
                        help="RL algorithm to use.")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes.")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode.")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs and model checkpoints.")
    args = parser.parse_args()
    if args.algo == "ppo":
        train_ppo(args)
    elif args.algo == "maddpg":
        train_maddpg(args)
    elif args.algo == "mappo":
        train_mappo(args)

if __name__ == "__main__":
    main()
