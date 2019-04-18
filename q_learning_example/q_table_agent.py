from environment import Environment
from q_agent import QAgent
import numpy as np


class QTableAgent(QAgent):
    def __init__(self, env: Environment, episodes: int) -> None:
        super().__init__(env, episodes)
        self.learning_rate = 0.03
        self.discount = 0.95
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
        print('{:>4} {:>4} {}'.format(
            observation,
            reward,
            table[new_ob]),
        )
        table[observation][action] *= (1 - self.learning_rate)
        table[observation][action] += self.learning_rate\
            * (reward + self.discount * future_reward)
