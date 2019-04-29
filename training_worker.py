import numpy as np
from typing import Tuple
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class TrainingWorker(QObject):
    done = pyqtSignal()
    output_ready = pyqtSignal(str, str)

    def __init__(self, env, agent, episode_count: int,
                 maximum_steps: int) -> None:
        super().__init__()
        self.env = env
        self.agent = agent
        self.episode_count = episode_count
        self.maximum_steps = maximum_steps
        self.running = True

    def __del__(self) -> None:
        self.wait()

    def _train_episode(self) -> Tuple[bool, int]:
        '''
        Train an episode and return:
            timed_out (bool): whether the training hasn't end before
             maximum_steps
            step_count (int): the step count when the episode end
            total_reward (int): the accumulated reward when the episode end
        '''
        observation = self.env.reset()
        total_reward = 0.0
        for step in range(self.maximum_steps):
            action = self.agent.get_action(observation)
            new_ob, reward, done = self.env.step(action)
            self.output_ready.emit(
                "Steps",
                '{:>2}\n{} ->\n{}; r = {:>5}'.format(
                    action,
                    observation,
                    new_ob,
                    reward,
                )
            )
            total_reward += reward
            self.agent.update(observation, new_ob, action, reward)
            observation = new_ob
            if done:
                return False, step + 1, total_reward
        return not True, self.maximum_steps, total_reward

    @pyqtSlot()
    def start(self) -> None:
        np.set_printoptions(precision=6, suppress=True)
        for episode in range(self.episode_count):
            if not self.running:
                break
            timed_out, step_count, total_reward = self._train_episode()
            self.output_ready.emit(
                "Episodes",
                "Episode {} {} after step {}.".format(
                    episode + 1,
                    "timed out" if timed_out else "end",
                    step_count,
                )
            )
            self.output_ready.emit(
                "Episodes",
                "Total Reward: {:>5}; Average Reward: {:>5.2f}"
                .format(total_reward, total_reward / step_count)
            )
            self.agent.next_episode()
        np.set_printoptions()
        self.done.emit()

    @pyqtSlot()
    def stop(self) -> None:
        self.running = False
