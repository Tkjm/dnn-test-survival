from environment import Environment
import numpy as np


class QTableAgent():
    def __init__(self, env: Environment, episodes: int) -> None:
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.learning_rate = 0.03
        self.discount = 0.95
        self.exploration_delta = 1.0 / episodes
        self.action_space = env.action_space
        self.reset()

    def reset(self) -> None:
        self.exploration_rate = 1.0

    def get_action(self, observation: int) -> int:
        if np.random.random() > self.exploration_rate:
            return self.__get_greedy_action(observation)
        else:
            return self.action_space.sample()

    def __get_greedy_action(self, observation: int) -> int:
        return np.random.choice(np.flatnonzero(
            self.q_table[observation] == self.q_table[observation].max()
        ))

    def update(
            self,
            observation: int,
            new_ob: int,
            action: int,
            reward: float,
    ) -> None:
        table = self.q_table
        future_reward = table[new_ob].max()
        print('{:>4} {:>4} {}'.format(
            observation,
            reward,
            table[new_ob]),
        )
        table[observation][action] *= (1 - self.learning_rate)
        table[observation][action] += self.learning_rate\
            * (reward + self.discount * future_reward)

    def next_episode(self) -> None:
        if self.exploration_rate > 0.0:
            self.exploration_rate -= self.exploration_delta
