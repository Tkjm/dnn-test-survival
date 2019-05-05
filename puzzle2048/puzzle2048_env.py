import gym
from gym.spaces import Discrete, Box
import numpy as np
from typing import Tuple
# from pdb import set_trace


class Puzzle2048_env(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self) -> None:
        self.width = 4
        self.height = 4
        self.action_space = Discrete(4)
        self.observation_space = Box(0.0, 1.0, (self.height, self.width))
        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        temp_map = np.rot90(self.map, action)
        reward = 0.0
        effective = False
        for col in range(self.width):
            row_merge = 0
            for row in range(1, self.height):
                if temp_map[row, col] == 0.0:
                    continue
                if temp_map[row, col] == temp_map[row_merge, col]:
                    effective = True
                    reward += temp_map[row_merge, col]
                    temp_map[row_merge, col] += 1.0
                    if temp_map[row_merge, col] == 9.0:
                        self.episode_over = True
                    temp_map[row, col] = 0.0
                else:
                    if temp_map[row_merge, col] != 0.0:
                        row_merge += 1
                    if row_merge != row:
                        effective = True
                        temp_map[row_merge, col], temp_map[row, col] =\
                            temp_map[row, col], temp_map[row_merge, col]
        self.map = np.rot90(temp_map, 4 - action)
        self.step_count += 1
        if effective:
            self.add_tile()
            zero_indices = np.argwhere(self.map == 0)
            if zero_indices.size == 0\
                and 0.0 not in np.diff(self.map)\
                    and 0.0 not in np.diff(self.map.T):
                self.episode_over = True
        return self.__get_observation(), reward, self.episode_over

    def reset(self) -> np.ndarray:
        self.map = np.zeros(shape=(self.height, self.width))
        self.episode_over = False
        self.step_count = 0
        self.add_tile()
        return self.__get_observation()

    def add_tile(self) -> None:
        zero_indices = np.argwhere(self.map == 0)
        if zero_indices.size > 0:
            choice = zero_indices[np.random.choice(zero_indices.shape[0])]
            self.map[tuple(choice)] = np.random.randint(1, 3)

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            # return RGB frame suitable for video
            raise NotImplementedError
        else:
            # just raise an exception
            super().render(mode=mode)

    def __get_observation(self) -> np.ndarray:
        observation = np.copy(self.map)
        return observation
