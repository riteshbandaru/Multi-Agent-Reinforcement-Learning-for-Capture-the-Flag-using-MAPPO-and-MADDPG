import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class MAPPOPolicy(nn.Module):
    """
    A simple MAPPO policy network that outputs both action logits and a state value.
    Observations are flattened.
    """
    def __init__(self, obs_shape, action_dim):
        super(MAPPOPolicy, self).__init__()
        flat_dim = int(np.prod(obs_shape))
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs):
        # obs shape: (batch, C, H, W) or flattened (batch, flat_dim)
        if len(obs.shape) > 2:
            x = obs.view(obs.size(0), -1)
        else:
            x = obs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)
        value = self.value_head(x)
        return logits, value

class MAPPOAgent:
    """
    MAPPO agent for a single agent.
    """
    def __init__(self, obs_shape, action_dim, lr=3e-4, gamma=0.99, clip_param=0.2,
                 vf_coef=0.5, ent_coef=0.01):
        self.policy = MAPPOPolicy(obs_shape, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_param = clip_param
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def select_action(self, obs):
        """
        Given a torch.Tensor observation with shape (1, C, H, W), sample an action.
        Returns: action (int), log probability (Tensor), value estimate (Tensor).
        """
        logits, value = self.policy(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        logits, value = self.policy(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, value, dist_entropy

    def update(self, trajectories):
        """
        Update MAPPO policy based on trajectories.
        trajectories: a list of dictionaries with keys: 
            'obs', 'action', 'log_prob', 'reward', 'value', 'done'
        For simplicity, we compute discounted returns in one pass.
        """
        obs = torch.cat([traj["obs"] for traj in trajectories])
        actions = torch.tensor([traj["action"] for traj in trajectories])
        old_log_probs = torch.cat([traj["log_prob"] for traj in trajectories])
        rewards = [traj["reward"] for traj in trajectories]
        dones = [traj["done"] for traj in trajectories]
        values = torch.cat([traj["value"] for traj in trajectories]).detach()

        # Compute discounted returns (simple method)
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        new_log_probs, new_values, entropy = self.evaluate_actions(obs, actions)
        advantages = returns - values.squeeze()
        ratios = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values.squeeze(), returns)
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
