from environment import Environment
import numpy as np
# import tensorflow as tf
import pdb


learning_rate = 0.1
discount = 0.95
exploration_rate = 1.0
exploration_delta = 1.0 / 200

env = Environment()
q_table = np.zeros((env.observation_space.n, env.action_space.n))

np.set_printoptions(precision=4)

for episode in range(200):
    observation = env.reset()
    print("Episode {} starts:".format(episode))
    done = False
    for step in range(200):
        action = None
        if np.random.random() > exploration_rate:
            action = np.random.choice(
                np.flatnonzero(
                    q_table[observation] == q_table[observation].max()
                )
            )
        else:
            action = env.action_space.sample()
        new_ob, reward, done = env.step(action)
        print('{:>4} {:>4} {}'.format(new_ob, reward, q_table[new_ob]))
        future_reward = q_table[new_ob][np.random.choice(
            np.flatnonzero(
                q_table[new_ob] == q_table[new_ob].max()
            )
        )]
        q_table[observation][action] += learning_rate\
            * (reward + discount * future_reward
               - q_table[observation][action])
        observation = new_ob
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break
    if not done:
        print("Episode timed out after {} timesteps".format(200))
    if exploration_rate > 0.0:
        exploration_rate -= exploration_delta

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(8, 8)),
#   tf.keras.layers.Dense(64, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(4, activation=tf.nn.softmax)
# ])
