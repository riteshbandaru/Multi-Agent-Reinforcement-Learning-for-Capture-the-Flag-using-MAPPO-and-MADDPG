import gymnasium as gym
from ctf_env import TeamEnhancedCaptureTheFlagEnv


class MultiAgentCTFWrapper(gym.Wrapper):
    """
    This wrapper converts the original environment to a multi-agent interface.
    Reset and step return a dictionary mapping agent names to their local observations.
    When receiving actions, it accepts a dictionary and converts it into a list in
    the order: ['red_0', 'red_1', 'blue_0', 'blue_1'].
    """
    def __init__(self, env):
        super().__init__(env)
        self.agent_ids = ["red_0", "red_1", "blue_0", "blue_1"]

        # Update observation space for partial views.
        vr = self.env.vision_range
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(2 * vr + 1, 2 * vr + 1, 3),
            dtype=self.env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        full_obs, info = self.env.reset(**kwargs)
        return self.env.get_agent_observations(), info

    def step(self, action_dict):
        # Convert action dict to list in fixed order
        actions = [action_dict[a] for a in self.agent_ids]
        full_obs, reward, done, truncated, info = self.env.step(actions)
        return self.env.get_agent_observations(), reward, done, truncated, info

    def render(self):
        return self.env.render()
