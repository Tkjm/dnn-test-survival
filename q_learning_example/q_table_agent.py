from .table_env import TableEnv
from .q_agent import QAgent

import numpy as np


class QTableAgent(QAgent):
    def __init__(self, env: TableEnv, episodes: int) -> None:
        super().__init__(env, episodes)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.q_table = np.zeros(
            (self.observation_space.n, self.action_space.n)
        )

    def _get_greedy_action(self, observation: int) -> int:
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
        learning_rate = 0.03
        table[observation][action] *= (1 - learning_rate)
        table[observation][action] += learning_rate\
            * (reward + self.discount * future_reward)
