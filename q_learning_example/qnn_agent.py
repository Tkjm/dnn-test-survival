from .table_env import TableEnv
from .q_agent import QAgent

import numpy as np
from tensorflow import keras


class QNNAgent(QAgent):
    def __init__(self, env: TableEnv, episodes: int) -> None:
        super().__init__(env, episodes)
        self.reset()

    def reset(self) -> None:
        super().reset()
        keras.backend.clear_session()
        self.model = keras.models.Sequential([
            keras.layers.Dense(
                self.action_space.n,
                input_shape=(self.observation_space.n,),
                kernel_initializer=keras.initializers.Zeros(),
                use_bias=False
            ),
        ])
        self.model.compile(
            optimizer=keras.optimizers.SGD(lr=0.1),
            loss=keras.losses.mean_squared_error,
        )

    def __get_q_value(self, observation: int) -> np.ndarray:
        return self.model.predict(
            np.eye(self.observation_space.n)[[observation]],
            batch_size=1,
        )[0]

    def _get_greedy_action(self, observation: int) -> int:
        qualities = self.__get_q_value(observation)
        return np.random.choice(np.flatnonzero(qualities == qualities.max()))

    def update(
            self,
            observation: int,
            new_ob: int,
            action: int,
            reward: float,
    ) -> None:
        new_q_value = self.__get_q_value(new_ob)
        desired_q_value = self.__get_q_value(observation)
        desired_q_value[action] = reward + self.discount * new_q_value.max()
        self.model.fit(
            x=np.eye(self.observation_space.n)[[observation]],
            y=desired_q_value[np.newaxis],
            batch_size=1,
            verbose=0,
        )
