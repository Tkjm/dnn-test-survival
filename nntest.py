from environment import Environment
import tensorflow as tf
import numpy as np
import pdb

env = Environment()
env.step(0)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(8, 8)),
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])
