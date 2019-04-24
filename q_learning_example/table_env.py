import gym
from gym.spaces import Discrete
import numpy as np
from typing import Tuple


class TableEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self) -> None:
        self.width = 3
        self.height = 3
        self.reward_pos = np.array([self.height - 1, self.width - 1])
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.height * self.width)
        self.reset()

    def step(self, action: int) -> Tuple[int, float, bool]:
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.width - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.height - 1:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        reward = 0.0
        if np.array_equal(self.agent_pos, self.reward_pos):
            reward = 1.0
            self.episode_over = True
        self.step_count += 1
        return self.__get_observation(), reward, self.episode_over

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.episode_over = False
        self.step_count = 0
        return self.__get_observation()

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            # return RGB frame suitable for video
            raise NotImplementedError
        else:
            # just raise an exception
            super().render(mode=mode)

    def __get_observation(self) -> int:
        return self.agent_pos[0] * self.width + self.agent_pos[1]
