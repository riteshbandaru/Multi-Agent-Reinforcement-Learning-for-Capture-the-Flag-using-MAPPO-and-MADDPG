import random
import numpy as np

class ReplayBuffer:
    """
    A replay buffer for multi-agent algorithms storing 7–tuples:
      (local_obs, full_obs, joint_action, reward, next_local_obs, next_full_obs, done)
    """
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, local_obs, full_obs, joint_action, reward, next_local_obs, next_full_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # Store as a 7-tuple.
        self.buffer[self.position] = (local_obs, full_obs, joint_action, reward, next_local_obs, next_full_obs, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        local_obs, full_obs, joint_action, rewards, next_local_obs, next_full_obs, dones = map(np.array, zip(*batch))
        return local_obs, full_obs, joint_action, rewards, next_local_obs, next_full_obs, dones

    def __len__(self):
        return len(self.buffer)
