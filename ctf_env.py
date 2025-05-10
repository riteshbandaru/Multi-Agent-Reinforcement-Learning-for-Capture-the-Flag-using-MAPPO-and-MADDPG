import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import sys
import time

class TeamEnhancedCaptureTheFlagEnv(gym.Env):
    """
    A 15x15 grid Capture-the-Flag game with a maze-like layout.
    Two teams (Red & Blue) each have two agents.
    Flags are at opposite corners; a team wins when one agent captures the opponent flag.
    Dynamic obstacles, fog of war for partial observability, terrain effects and bonus items add complexity.
    
    Reward Scheme (per agent):
      - Primary objective: +10 for capturing the opponent flag.
      - Step penalty: –0.5 per step.
      - Terrain penalty: –1 for terrain cells.
      - Bonus items: +2 when collected.
      - Progress reward: +0.5 for each unit of Manhattan distance reduction.
      - Cooperative bonus: +1 to the team reward if the average progress exceeds a threshold.
      - Vision bonus: +0.5 if an agent sees an opponent in its local view.
    Team reward = sum of agents' rewards.
    
    Supports Gymnasium's render_mode API.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, vision_range=3):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = 15
        self.vision_range = vision_range  # radius for agent local observation

        # Flag positions: red flag is at (0, 0) and blue flag is at (14, 14)
        self.flag_red = (0, 0)         # Blue team must capture red flag.
        self.flag_blue = (14, 14)      # Red team must capture blue flag.

        # Agents: Two per team.
        self.agents_red = [[1, 1], [1, 2]]
        self.agents_blue = [[13, 13], [13, 12]]

        # Full observation space: 15x15 grid with 3 channels.
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.float32
        )

        # MultiDiscrete action space for four agents: [5,5,5,5]
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])

        # Reward parameters
        self.capture_reward = 10.0             # Primary objective (+10) for capturing flag
        self.step_penalty = -0.5               # Step penalty changed to -0.5 per step
        self.terrain_penalty = -1.0            # Terrain penalty remains -1
        self.bonus_reward = 2.0                # Bonus for bonus items set to +2
        self.progress_coef = 0.5               # Progress shaping coefficient: +0.5 per unit distance reduction
        self.vision_bonus = 0.5                # Vision bonus if an opponent is seen in local view
        self.coop_threshold = 0.5              # Threshold for average progress improvement (in Manhattan units)
        self.coop_bonus = 1.0                  # Cooperative bonus awarded when threshold is exceeded

        # Maze-like static obstacles layout
        self.static_obstacles = self._create_maze_layout()

        # Dynamic obstacles (updated every few steps)
        self.dynamic_obstacles = []

        # Terrain cells (penalty regions)
        self.terrain = [(5, 5), (5, 9), (9, 5), (9, 9)]

        # Bonus items: dictionary {position: bonus_reward}
        self.bonus_items = {}

        # Dynamic update parameters
        self.step_count = 0
        self.dynamic_move_frequency = 5      # update every 5 steps
        self.bonus_spawn_probability = 0.2

        # Initialize PyGame window if rendering is enabled.
        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 40
            self.scoreboard_height = 50
            self.width = self.grid_size * self.cell_size
            self.height = self.grid_size * self.cell_size + self.scoreboard_height
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Team Enhanced Capture The Flag - Maze Layout")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20, bold=True)

    def _create_maze_layout(self):
        """Create a maze layout with corridors for agent movement."""
        maze = []
        for i in range(self.grid_size):
            if i not in [0, 3, 7, 11, 14]:
                maze.append((0, i))
                maze.append((self.grid_size - 1, i))
                maze.append((i, 0))
                maze.append((i, self.grid_size - 1))
        maze.extend([(3, 2), (2, 4), (2, 10), (4, 12),
                     (10, 2), (8, 4), (10, 10), (12, 11),
                     (7, 3), (7, 11), (4, 7), (10, 7),
                     (3, 3), (3, 11), (11, 3), (11, 11),
                     (5, 6), (9, 8), (4, 4), (10, 10)])
        exclusion = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2),
                     (14, 14), (14, 13), (13, 14), (13, 13), (13, 12)]
        maze = [p for p in maze if p not in exclusion]
        return maze

    def _manhattan_distance(self, pos1, pos2):
        """Return Manhattan distance between pos1 and pos2."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self, seed=None, options=None):
        self.agents_red = [[1, 1], [1, 2]]
        self.agents_blue = [[13, 13], [13, 12]]
        self.step_count = 0
        self.dynamic_obstacles = self._generate_dynamic_obstacles(n=2)
        self.bonus_items = {}
        observation = self.get_observation()  # full observation
        return observation, {}

    def step(self, actions):
        # Compute current Manhattan distances for red and blue agents to target flags.
        red_distances = [self._manhattan_distance(agent, self.flag_blue) for agent in self.agents_red]
        blue_distances = [self._manhattan_distance(agent, self.flag_red) for agent in self.agents_blue]

        # Expect actions as a list: [action_red0, action_red1, action_blue0, action_blue1]
        a_red1, a_red2, a_blue1, a_blue2 = map(int, actions)
        self.agents_red[0] = self._move_agent(self.agents_red[0], a_red1)
        self.agents_red[1] = self._move_agent(self.agents_red[1], a_red2)
        self.agents_blue[0] = self._move_agent(self.agents_blue[0], a_blue1)
        self.agents_blue[1] = self._move_agent(self.agents_blue[1], a_blue2)

        # Calculate rewards for red agents, including progress reward and vision bonus.
        rewards_red = []
        individual_progress = []
        for i, pos in enumerate(self.agents_red):
            new_distance = self._manhattan_distance(pos, self.flag_blue)
            progress_reward = self.progress_coef * (red_distances[i] - new_distance)
            individual_progress.append(red_distances[i] - new_distance)
            if pos == list(self.flag_blue):
                r = self.capture_reward
            else:
                r = self.step_penalty + progress_reward
            if tuple(pos) in self.terrain:
                r += self.terrain_penalty
            if tuple(pos) in self.bonus_items:
                r += self.bonus_reward
                del self.bonus_items[tuple(pos)]
            # Vision bonus for red agent: if local observation in blue channel has at least one 1.
            local_obs = self.get_local_observation(pos)
            if np.any(local_obs[:, :, 1] == 1.0):
                r += self.vision_bonus
            rewards_red.append(r)
        
        # Calculate rewards for blue agents similarly.
        rewards_blue = []
        for i, pos in enumerate(self.agents_blue):
            new_distance = self._manhattan_distance(pos, self.flag_red)
            progress_reward = self.progress_coef * (blue_distances[i] - new_distance)
            if pos == list(self.flag_red):
                r = self.capture_reward
            else:
                r = self.step_penalty + progress_reward
            if tuple(pos) in self.terrain:
                r += self.terrain_penalty
            if tuple(pos) in self.bonus_items:
                r += self.bonus_reward
                del self.bonus_items[tuple(pos)]
            # Vision bonus for blue agent: check channel 0.
            local_obs = self.get_local_observation(pos)
            if np.any(local_obs[:, :, 0] == 1.0):
                r += self.vision_bonus
            rewards_blue.append(r)
        
        # Compute team rewards (for training, we might use red team's reward).
        total_r_red = sum(rewards_red)
        total_r_blue = sum(rewards_blue)
        
        # Cooperative bonus for red team: if the average progress improvement is above a threshold,
        # add a bonus.
        avg_progress_red = np.mean(individual_progress)  # average improvement for red agents
        if avg_progress_red > self.coop_threshold:
            total_r_red += self.coop_bonus

        self.step_count += 1
        if self.step_count % self.dynamic_move_frequency == 0:
            self.dynamic_obstacles = self._generate_dynamic_obstacles(n=2)
        if random.random() < self.bonus_spawn_probability:
            self._spawn_bonus_item()

        done = (any(pos == list(self.flag_blue) for pos in self.agents_red) or
                any(pos == list(self.flag_red) for pos in self.agents_blue))
        # For training (e.g., PPO), return the red team's total reward.
        return self.get_observation(), total_r_red, done, False, {}

    def _move_agent(self, agent, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        new_pos = [agent[0] + moves[action][0], agent[1] + moves[action][1]]
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return agent
        if tuple(new_pos) in self.static_obstacles or tuple(new_pos) in self.dynamic_obstacles:
            return agent
        return new_pos

    def _generate_dynamic_obstacles(self, n=2):
        blocked = (self.static_obstacles + self.terrain +
                   [tuple(a) for a in (self.agents_red + self.agents_blue)] +
                   [self.flag_red, self.flag_blue])
        candidates = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                      if (r, c) not in blocked]
        if len(candidates) < n:
            return []
        return random.sample(candidates, n)

    def _spawn_bonus_item(self):
        blocked = (self.static_obstacles + self.dynamic_obstacles + self.terrain +
                   list(self.bonus_items.keys()) +
                   [tuple(a) for a in (self.agents_red + self.agents_blue)] +
                   [self.flag_red, self.flag_blue])
        candidates = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                      if (r, c) not in blocked]
        if candidates:
            pos = random.choice(candidates)
            self.bonus_items[pos] = self.bonus_reward

    def get_observation(self):
        """Return the full observation grid with three channels."""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        # Red agents in channel 0
        for pos in self.agents_red:
            obs[pos[0], pos[1], 0] = 1.0
        # Blue agents in channel 1
        for pos in self.agents_blue:
            obs[pos[0], pos[1], 1] = 1.0
        # Flags and bonus items in channel 2
        obs[self.flag_red[0], self.flag_red[1], 2] = 1.0
        obs[self.flag_blue[0], self.flag_blue[1], 2] = 1.0
        for pos in self.bonus_items.keys():
            obs[pos[0], pos[1], 2] = 0.8
        return obs

    def get_local_observation(self, pos):
        """
        Return a partial view (fog-of-war) observation around the given agent position.
        The view is (2*vision_range+1)x(2*vision_range+1)x3. Out-of-bound areas are zero-padded.
        """
        full_obs = self.get_observation()
        r, c = pos
        vr = self.vision_range
        obs_local = np.zeros((2 * vr + 1, 2 * vr + 1, 3), dtype=np.float32)
        for i in range(-vr, vr + 1):
            for j in range(-vr, vr + 1):
                rr = r + i
                cc = c + j
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    obs_local[i + vr, j + vr] = full_obs[rr, cc]
        return obs_local

    def get_agent_observations(self):
        """
        Return a dictionary of local observations for each agent.
        Keys: 'red_0', 'red_1', 'blue_0', 'blue_1'
        Each observation is (2*vision_range+1)x(2*vision_range+1)x3.
        """
        obs_dict = {}
        obs_dict["red_0"] = self.get_local_observation(self.agents_red[0])
        obs_dict["red_1"] = self.get_local_observation(self.agents_red[1])
        obs_dict["blue_0"] = self.get_local_observation(self.agents_blue[0])
        obs_dict["blue_1"] = self.get_local_observation(self.agents_blue[1])
        return obs_dict

    def render(self):
        if self.render_mode != "human":
            return None
        self.screen.fill((60, 60, 60))
        grid_offset_y = self.scoreboard_height

        # Draw floor (checkerboard)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x_pix = col * self.cell_size
                y_pix = row * self.cell_size + grid_offset_y
                color = (240, 224, 208) if (row + col) % 2 == 0 else (220, 204, 188)
                rect = pygame.Rect(x_pix, y_pix, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)

        # Apply fog-of-war: mark unseen areas as dark.
        visibility = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for pos in self.agents_red + self.agents_blue:
            r, c = pos
            vr = self.vision_range
            for i in range(max(0, r - vr), min(self.grid_size, r + vr + 1)):
                for j in range(max(0, c - vr), min(self.grid_size, c + vr + 1)):
                    visibility[i, j] = True

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if not visibility[row, col]:
                    x_pix = col * self.cell_size
                    y_pix = row * self.cell_size + grid_offset_y
                    fog_rect = pygame.Surface((self.cell_size, self.cell_size))
                    fog_rect.set_alpha(150)
                    fog_rect.fill((30, 30, 30))
                    self.screen.blit(fog_rect, (x_pix, y_pix))

        # Draw static obstacles (walls)
        for pos in self.static_obstacles:
            x_pix = pos[1] * self.cell_size
            y_pix = pos[0] * self.cell_size + grid_offset_y
            pygame.draw.rect(self.screen, (160, 82, 45), (x_pix, y_pix, self.cell_size, self.cell_size))

        # Draw dynamic obstacles
        for pos in self.dynamic_obstacles:
            x_pix = pos[1] * self.cell_size
            y_pix = pos[0] * self.cell_size + grid_offset_y
            pygame.draw.rect(self.screen, (140, 150, 200), (x_pix, y_pix, self.cell_size, self.cell_size))

        # Draw terrain cells
        for pos in self.terrain:
            x_pix = pos[1] * self.cell_size
            y_pix = pos[0] * self.cell_size + grid_offset_y
            pygame.draw.rect(self.screen, (139, 69, 19), (x_pix, y_pix, self.cell_size, self.cell_size))

        # Draw bonus items
        for pos in self.bonus_items.keys():
            x_pix = pos[1] * self.cell_size + self.cell_size // 2
            y_pix = pos[0] * self.cell_size + grid_offset_y + self.cell_size // 2
            pygame.draw.circle(self.screen, (255, 223, 0), (x_pix, y_pix), self.cell_size // 4)

        # Draw flags
        pygame.draw.rect(self.screen, (255, 50, 50),
                         (self.flag_red[1] * self.cell_size,
                          self.flag_red[0] * self.cell_size + grid_offset_y,
                          self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (100, 150, 255),
                         (self.flag_blue[1] * self.cell_size,
                          self.flag_blue[0] * self.cell_size + grid_offset_y,
                          self.cell_size, self.cell_size))

        # Draw Red agents
        for pos in self.agents_red:
            x_pix = pos[1] * self.cell_size + self.cell_size // 2
            y_pix = pos[0] * self.cell_size + grid_offset_y + self.cell_size // 2
            pygame.draw.circle(self.screen, (255, 99, 71), (x_pix, y_pix), self.cell_size // 3)

        # Draw Blue agents
        for pos in self.agents_blue:
            x_pix = pos[1] * self.cell_size + self.cell_size // 2
            y_pix = pos[0] * self.cell_size + grid_offset_y + self.cell_size // 2
            pygame.draw.circle(self.screen, (65, 105, 225), (x_pix, y_pix), self.cell_size // 3)

        # Draw scoreboard
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.width, self.scoreboard_height))
        score_text = "Team Rewards (Placeholder)"
        text_surf = self.font.render(score_text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(self.width // 2, self.scoreboard_height // 2))
        self.screen.blit(text_surf, text_rect)

        pygame.display.flip()
        self.clock.tick(5)


if __name__ == "__main__":
    env = TeamEnhancedCaptureTheFlagEnv(render_mode="human")
    obs, _ = env.reset()
    print("Observation shape:", obs.shape)
    for _ in range(5):
        actions = env.action_space.sample()
        obs, reward, done, _, _ = env.step(actions)
        print("Actions:", actions, "Reward:", reward, "Done:", done)
        env.render()
        if done:
            break
    time.sleep(2)
    pygame.quit()
    sys.exit()
