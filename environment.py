import numpy as np
import pdb
from typing import Tuple
from gym.spaces import Discrete


class Environment():
    def __init__(self) -> None:
        self.map = np.ndarray(shape=(8, 8))
        self.reward_pos = np.array([7, 7])
        self.action_space = Discrete(4)
        self.observation_space = Discrete(64)
        self.reset()

    def step(self, action: int) -> Tuple[int, float, bool]:
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < 7:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < 7:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        reward = 0.0
        if np.array_equal(self.agent_pos, self.reward_pos):
            reward = 1.0
            self.episode_over = True
        return self.__get_observation(), reward, self.episode_over

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.episode_over = False
        return self.__get_observation()

    def __get_observation(self) -> int:
        return self.agent_pos[0] * 8 + self.agent_pos[1]
