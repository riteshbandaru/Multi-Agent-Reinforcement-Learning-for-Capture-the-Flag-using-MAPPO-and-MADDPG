import argparse
import sys
import time
import torch
import pygame
from ctf_env import TeamEnhancedCaptureTheFlagEnv
from wrappers import MultiAgentCTFWrapper

def play_maddpg(args):
    base_env = TeamEnhancedCaptureTheFlagEnv(render_mode="human")
    env = MultiAgentCTFWrapper(base_env)
    
    ckpt = torch.load(args.model_checkpoint)
    from maddpg import MADDPG
    agent_ids = ["red_0", "red_1", "blue_0", "blue_1"]
    obs_dim = (2 * env.env.vision_range + 1) ** 2 * 3
    action_dim = 5
    full_obs_dim = env.env.grid_size * env.env.grid_size * 3
    joint_action_dim = action_dim * 4
    maddpg = MADDPG(obs_dim, action_dim, full_obs_dim, joint_action_dim, num_agents=4, use_commnet=True)
    
    for i in range(4):
        maddpg.agents[i].actor.load_state_dict(ckpt["agent_" + str(i)])
        maddpg.agents[i].actor.eval()
    
    done = False
    while not done:
        obs_dict = env.env.get_agent_observations()
        local_obs_batch = [obs_dict[aid].flatten() for aid in agent_ids]
        actions = maddpg.act(local_obs_batch, noise=0.0)
        actions_dict = {agent_ids[i]: actions[i] for i in range(len(agent_ids))}
        obs, reward, done, _, _ = env.step(actions_dict)
        env.render()
        time.sleep(0.2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
    time.sleep(2)

def play_ppo(args):
    from ppo import PPOAgent
    base_env = TeamEnhancedCaptureTheFlagEnv(render_mode="human")
    env = MultiAgentCTFWrapper(base_env)
    ckpt = torch.load(args.model_checkpoint)
    agent_ids = ["red_0", "red_1", "blue_0", "blue_1"]
    ppo_agents = {}
    obs_shape = env.observation_space.shape
    for aid in agent_ids:
        agent = PPOAgent(obs_shape, action_dim=5)
        agent.policy.load_state_dict(ckpt[aid])
        agent.policy.eval()
        ppo_agents[aid] = agent
    done = False
    while not done:
        obs_dict = env.env.get_agent_observations()
        actions = {}
        for aid in agent_ids:
            obs = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)
            action, _, _ = ppo_agents[aid].select_action(obs)
            actions[aid] = action
        obs_dict, reward, done, _, _ = env.step(actions)
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        time.sleep(0.2)
    time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="maddpg", choices=["ppo", "maddpg"],
                        help="RL algorithm used.")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the saved model checkpoint.")
    args = parser.parse_args()
    if args.algo == "maddpg":
        play_maddpg(args)
    elif args.algo == "ppo":
        play_ppo(args)

if __name__ == "__main__":
    main()
