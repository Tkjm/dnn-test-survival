from table_env import TableEnv
from nn_q_agent import NNQAgent
# from q_table_agent import QTableAgent
import pdb
import numpy as np
import os

USE_CPU = True
if (USE_CPU):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

episode_count = 200
maximum_steps = 600
env = TableEnv()
agent = NNQAgent(env, episodes=episode_count)
# agent = QTableAgent(env, episodes=episode_count)

np.set_printoptions(precision=6, suppress=True)

for episode in range(episode_count):
    observation = env.reset()
    print("Episode {} starts:".format(episode))
    done = False
    for step in range(maximum_steps):
        action = agent.get_action(observation)
        new_ob, reward, done = env.step(action)
        agent.update(observation, new_ob, action, reward)
        observation = new_ob
        if done:
            print("Episode finished after {} timesteps".format(step + 1))
            break
    if not done:
        print("Episode timed out after {} timesteps".format(maximum_steps))
    agent.next_episode()
# print('Fianl Q Table \n{}'.format(agent.q_table))
