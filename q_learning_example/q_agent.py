from environment import Environment
import numpy as np


class QAgent():
    def __init__(self, env: Environment, episodes: int) -> None:
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.exploration_delta = 1.0 / episodes

    def get_action(self, observation: int) -> int:
        if np.random.random() > self.exploration_rate:
            return self._get_greedy_action(observation)
        else:
            return self.action_space.sample()

    def _get_greedy_action(self, observation: int) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        self.exploration_rate = 1.0

    def update(
            self,
            observation: int,
            new_ob: int,
            action: int,
            reward: float,
    ) -> None:
        raise NotImplementedError

    def next_episode(self) -> None:
        if self.exploration_rate > 0.0:
            self.exploration_rate -= self.exploration_delta
