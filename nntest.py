from environment import Environment
# import tensorflow as tf
import numpy as np
import pdb

env = Environment()

for i_episode in range(20):
    observation = env.reset()
    print("Episode Start")
    for t in range(100):
        print(observation)
        observation, reward, done = env.step(np.random.randint(0, 3))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(8, 8)),
#   tf.keras.layers.Dense(64, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(4, activation=tf.nn.softmax)
# ])
