import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOPolicy(nn.Module):
    """
    A simple policy network for PPO.
    This network will output logits over actions for an agent.
    """
    def __init__(self, obs_shape, action_dim):
        super(PPOPolicy, self).__init__()
        flat_dim = np.prod(obs_shape)
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs):
        # Flatten the observation.
        x = obs.view(obs.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Actor: logits for discrete actions.
        logits = self.action_head(x)
        # Critic: state value.
        value = self.value_head(x)
        return logits, value

class PPOAgent:
    """
    PPO agent to be used for multi-agent training.
    In this simple implementation, we assume a decentralized setup: one network per agent.
    """
    def __init__(self, obs_shape, action_dim, lr=1e-3, gamma=0.99, clip_param=0.2, entropy_coef=0.01):
        self.policy = PPOPolicy(obs_shape, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef

    def select_action(self, obs):
        """
        Select action using the current policy.
        obs: a torch.Tensor of shape (1, C, H, W) e.g. partial observation.
        Returns: action (int) and log probability.
        """
        logits, value = self.policy(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def update(self, trajectories):
        """
        Update the PPO policy using trajectories.
        trajectories: a list of dictionaries containing keys 'obs', 'action', 'log_prob', 'reward', 'value', 'done'
        """
        obs = torch.cat([traj["obs"] for traj in trajectories])
        actions = torch.tensor([traj["action"] for traj in trajectories])
        old_log_probs = torch.cat([traj["log_prob"] for traj in trajectories])
        rewards = [traj["reward"] for traj in trajectories]
        dones = [traj["done"] for traj in trajectories]
        values = torch.cat([traj["value"] for traj in trajectories]).detach()

        # Compute discounted returns
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits, new_values = self.policy(obs)
        new_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Ratio for PPO clipping
        ratios = torch.exp(new_log_probs - old_log_probs.detach())
        advantages = returns - values.squeeze()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# For multi-agent PPO, you can instantiate one PPOAgent per agent.
if __name__ == "__main__":
    # Test a PPOAgent with a dummy observation.
    obs_shape = (3, 7, 7)  # for vision_range=3 -> (2*3+1=7)x7 with 3 channels
    action_dim = 5  # 5 discrete actions
    agent = PPOAgent(obs_shape, action_dim)
    dummy_obs = torch.randn(1, *obs_shape)
    action, log_prob, value = agent.select_action(dummy_obs)
    print("Action:", action, "Value:", value.item())
