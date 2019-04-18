from gym.spaces import Discrete
import numpy as np
import pdb
from typing import Tuple


class Environment():
    def __init__(self) -> None:
        self.width = 3
        self.height = 3
        self.map = np.ndarray(shape=(self.height, self.width))
        self.reward_pos = np.array([self.height - 1, self.width - 1])
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.height * self.width)
        self.reset()

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
        """
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
            reward = 1
            self.episode_over = True
        self.step_count += 1
        return self.__get_observation(), reward, self.episode_over

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.episode_over = False
        self.step_count = 0
        return self.__get_observation()

    def __get_observation(self) -> int:
        return self.agent_pos[0] * self.width + self.agent_pos[1]
