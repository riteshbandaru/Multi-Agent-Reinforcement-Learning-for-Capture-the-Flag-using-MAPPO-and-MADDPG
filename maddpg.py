import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from buffer import ReplayBuffer
from commnet import CommNet  # Our communication network module

# Standard Actor and CommNet-enabled Actor (as defined earlier)
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class CommNetActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(CommNetActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    def forward(self, obs, comm_message):
        x = F.relu(self.fc1(obs))
        x = x + comm_message   # Incorporate communication
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, full_obs_dim, full_action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(full_obs_dim + full_action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, full_obs_dim, full_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, index=0, use_commnet=False):
        self.index = index
        self.use_commnet = use_commnet
        if self.use_commnet:
            self.actor = CommNetActor(obs_dim, action_dim, hidden_dim=128)
            self.target_actor = CommNetActor(obs_dim, action_dim, hidden_dim=128)
        else:
            self.actor = Actor(obs_dim, action_dim)
            self.target_actor = Actor(obs_dim, action_dim)
        self.critic = Critic(full_obs_dim, full_action_dim)
        self.target_critic = Critic(full_obs_dim, full_action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)
        
    def _hard_update(self, target, source):
        target.load_state_dict(source.state_dict())
        
    def act(self, obs, comm_message=None, noise=0.0):
        obs_tensor = torch.FloatTensor(obs.reshape(1, -1))
        if self.use_commnet:
            if comm_message is None:
                raise ValueError("Comm message is required when using CommNet.")
            comm_tensor = torch.FloatTensor(comm_message.reshape(1, -1))
            action_probs = self.actor(obs_tensor, comm_tensor).detach().numpy().flatten()
        else:
            action_probs = self.actor(obs_tensor).detach().numpy().flatten()
        if noise > 0.0:
            action_probs += noise * np.random.randn(*action_probs.shape)
            action_probs = np.clip(action_probs, 0, 1)
        return np.argmax(action_probs)
    
    def update(self, samples, gamma=0.99, tau=0.01):
        local_obs_all, full_obs, joint_actions, rewards, next_local_obs_all, next_full_obs, dones = samples
        local_obs = torch.FloatTensor([sample[self.index] for sample in local_obs_all])
        next_local_obs = torch.FloatTensor([sample[self.index] for sample in next_local_obs_all])
        full_obs = torch.FloatTensor(full_obs)
        joint_actions = torch.FloatTensor(joint_actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_full_obs = torch.FloatTensor(next_full_obs)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        action_dim = self.actor.fc3.out_features
        
        # Compute target Q
        if self.use_commnet:
            zeros = torch.zeros(next_local_obs.shape[0], 128)
            next_action = self.target_actor(next_local_obs, zeros)
        else:
            next_action = self.target_actor(next_local_obs)
        joint_next_actions = joint_actions.clone()
        start = self.index * action_dim
        end = (self.index + 1) * action_dim
        joint_next_actions[:, start:end] = next_action
        target_q = self.target_critic(next_full_obs, joint_next_actions)
        y = rewards + gamma * target_q * (1 - dones)
        
        q = self.critic(full_obs, joint_actions)
        critic_loss = F.mse_loss(q, y.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        if self.use_commnet:
            zeros = torch.zeros(local_obs.shape[0], 128)
            current_action = self.actor(local_obs, zeros)
        else:
            current_action = self.actor(local_obs)
        joint_current_actions = joint_actions.clone()
        joint_current_actions[:, start:end] = current_action
        actor_loss = -self.critic(full_obs, joint_current_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)
        
        return actor_loss.item(), critic_loss.item()
    
    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

class MADDPG:
    def __init__(self, obs_dim, action_dim, full_obs_dim, joint_action_dim, num_agents=4, use_commnet=False):
        self.num_agents = num_agents
        self.use_commnet = use_commnet
        if self.use_commnet:
            self.commnet = CommNet(input_dim=obs_dim, hidden_dim=128, message_dim=128, n_agents=num_agents, n_layers=2)
        self.agents = [MADDPGAgent(obs_dim, action_dim, full_obs_dim, joint_action_dim, index=i, use_commnet=use_commnet)
                       for i in range(num_agents)]
        self.buffer = ReplayBuffer(capacity=100000)
        
    def act(self, obs_batch, noise=0.0):
        actions = []
        comm_messages = None
        if self.use_commnet:
            obs_tensor = torch.FloatTensor(np.stack(obs_batch, axis=0))
            comm_messages_tensor = self.commnet(obs_tensor)
            comm_messages = comm_messages_tensor.detach().numpy()
        for i in range(self.num_agents):
            msg = comm_messages[i] if self.use_commnet else None
            action = self.agents[i].act(obs_batch[i], comm_message=msg, noise=noise)
            actions.append(action)
        return actions
    
    def push(self, local_obs, full_obs, joint_action, reward, next_local_obs, next_full_obs, done):
        self.buffer.push(local_obs, full_obs, joint_action, reward, next_local_obs, next_full_obs, done)
        
    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return None
        samples = self.buffer.sample(batch_size)
        local_obs, full_obs, joint_actions, rewards, next_local_obs, next_full_obs, dones = samples
        losses = []
        for agent in self.agents:
            loss = agent.update((local_obs, full_obs, joint_actions, rewards, next_local_obs, next_full_obs, dones))
            losses.append(loss)
        return np.mean(losses, axis=0)
