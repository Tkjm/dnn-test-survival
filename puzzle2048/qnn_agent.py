from .puzzle2048_env import Puzzle2048_env
from .q_agent import QAgent

import numpy as np
from tensorflow import keras
import tensorflow as tf
# from pdb import set_trace


class QNNAgent(QAgent):
    def __init__(self, env: Puzzle2048_env, episodes: int) -> None:
        super().__init__(env, episodes)
        self.reset()

    def reset(self) -> None:
        super().reset()
        keras.backend.clear_session()
        self.model = keras.models.Sequential([
            keras.layers.Reshape(
                (*self.observation_space.shape, 1),
                input_shape=self.observation_space.shape,
            ),
            keras.layers.Conv2D(
                4,
                kernel_size=3,
                padding='same',
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(
                9,
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(
                7,
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(
                self.action_space.n,
            ),
        ])
        self.model.compile(
            optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.6),
            loss=keras.losses.mean_squared_error,
        )
        self.graph = tf.get_default_graph()

    def __get_q_value(self, observation: np.ndarray) -> np.ndarray:
        with self.graph.as_default():
            return self.model.predict(
                observation[np.newaxis, :],
                batch_size=1,
            )[0]

    def _get_greedy_action(self, observation: np.ndarray) -> int:
        qualities = self.__get_q_value(observation)
        max = qualities.max()
        if np.isnan(max):
            return self.action_space.sample()
        return np.random.choice(np.flatnonzero(qualities == max))

    def update(
            self,
            observation: np.ndarray,
            new_ob: int,
            action: int,
            reward: float,
    ) -> None:
        new_q_value = self.__get_q_value(new_ob)
        desired_q_value = self.__get_q_value(observation)
        desired_q_value[action] = reward + self.discount * new_q_value.max()
        with self.graph.as_default():
            self.model.fit(
                x=observation[np.newaxis, :],
                y=desired_q_value[np.newaxis],
                batch_size=1,
                verbose=0,
            )
