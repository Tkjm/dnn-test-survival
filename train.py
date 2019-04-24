import numpy as np
import os
from typing import Tuple, Callable


def train_episode(env, agent, maximum_steps: int,
                  outputFunc: Callable) -> Tuple[bool, int]:
    '''
    Train an episode and return:
        timed_out (bool): whether the training hasn't end before maximum_steps
        step_count (int): the step count when the episode end
        total_reward (int): the accumulated reward when the episode end
    '''
    observation = env.reset()
    total_reward = 0.0
    for step in range(maximum_steps):
        action = agent.get_action(observation)
        new_ob, reward, done = env.step(action)
        outputFunc('{:>2}\n{} ->\n{}; r = {:>5}'.format(
            action,
            observation,
            new_ob,
            reward,
        ))
        total_reward += reward
        agent.update(observation, new_ob, action, reward)
        observation = new_ob
        if done:
            return False, step + 1, total_reward
    return not True, maximum_steps, total_reward


def train(env, agent, episode_count: int, maximum_steps: int,
          outputFunc: Callable = print, use_cpu: bool = False):
    if (use_cpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.set_printoptions(precision=6, suppress=True)
    for episode in range(episode_count):
        timed_out, step_count, total_reward = train_episode(
            env, agent, maximum_steps, outputFunc)
        outputFunc("Episode {} {} after step {}.".format(
            episode + 1,
            "timed out" if timed_out else "end",
            step_count,
        ))
        outputFunc("Total Reward: {:>5}".format(total_reward))
        agent.next_episode()
    np.set_printoptions()
