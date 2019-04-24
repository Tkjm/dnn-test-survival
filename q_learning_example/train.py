from .table_env import TableEnv
from .qnn_agent import QNNAgent
from .q_table_agent import QTableAgent

import pdb
import numpy as np
import os
from typing import Tuple, Callable


def train_episode(env, agent, maximum_steps: int,
                  outputFunc: Callable) -> Tuple[bool, int]:
    '''
    Train an episode and return:
        timed_out (bool): whether the training hasn't end before maximum_steps
        step_count (int): the step count when the episode end
    '''
    observation = env.reset()
    for step in range(maximum_steps):
        action = agent.get_action(observation)
        new_ob, reward, done = env.step(action)
        outputFunc('{:>2}; {:>2} -> {:>2}; r = {:>3}'.format(
            action,
            observation,
            new_ob,
            reward,
        ))
        agent.update(observation, new_ob, action, reward)
        observation = new_ob
        if done:
            return False, step + 1
    return not True, maximum_steps


def train(outputFunc: Callable = print, use_cpu: bool = False):
    USE_NN = True

    if (use_cpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env = TableEnv()
    episode_count = 200
    maximum_steps = 600

    if USE_NN:
        agent = QNNAgent(env, episodes=episode_count)
    else:
        agent = QTableAgent(env, episodes=episode_count)

    np.set_printoptions(precision=6, suppress=True)
    for episode in range(episode_count):
        timed_out, step_count = train_episode(
            env, agent, maximum_steps, outputFunc)
        outputFunc("Episode {} {} after step {}.".format(
            episode,
            "timed out" if timed_out else "end",
            step_count,
        ))
        agent.next_episode()
    np.set_printoptions()
