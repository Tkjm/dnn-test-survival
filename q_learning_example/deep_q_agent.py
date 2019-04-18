from environment import Environment
import numpy as np
import tensorflow as tf
import pdb
from tensorflow.keras import layers


class DeepQAgent():
    def __init__(self, env: Environment, episodes: int) -> None:
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.discount = 0.95
        self.exploration_delta = 1.0 / episodes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.model = tf.keras.models.Sequential([
            layers.Dense(
                env.action_space.n,
                input_shape=(env.observation_space.n,),
                kernel_initializer='zeros',
                use_bias=False
            ),
        ])
        self.model.compile(
            optimizer=tf.train.GradientDescentOptimizer(0.1),
            loss='mean_squared_error',
        )
        self.reset()

    def reset(self) -> None:
        self.exploration_rate = 1.0

    def get_action(self, observation: int) -> int:
        if np.random.random() > self.exploration_rate:
            return self.__get_greedy_action(observation)
        else:
            return self.action_space.sample()

    def __get_q_value(self, observation: int) -> np.ndarray:
        return self.model.predict(
            np.eye(self.observation_space.n)[[observation]],
            batch_size=1,
        )[0]

    def __get_greedy_action(self, observation: int) -> int:
        qualities = self.__get_q_value(observation)
        return np.random.choice(np.flatnonzero(
            qualities == qualities.max()
        ))

    def update(
            self,
            observation: int,
            new_ob: int,
            action: int,
            reward: float,
    ) -> None:
        new_q_value = self.__get_q_value(new_ob)
        desired_q_value = self.__get_q_value(observation)
        print('{:>4} {:>4} {}'.format(observation, reward, desired_q_value))
        desired_q_value[action] = reward + self.discount * new_q_value.max()
        self.model.fit(
            x=np.eye(self.observation_space.n)[[observation]],
            y=desired_q_value[np.newaxis],
            batch_size=1,
            verbose=0,
        )

    def next_episode(self) -> None:
        if self.exploration_rate > 0.0:
            self.exploration_rate -= self.exploration_delta
