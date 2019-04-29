from q_learning_example.table_env import TableEnv
from q_learning_example.qnn_agent import QNNAgent
from q_learning_example.q_table_agent import QTableAgent
from main_app import MainApp

from training_worker import TrainingWorker
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
env = TableEnv()
USE_NN = True
episode_count = 200
maximum_steps = 600
if USE_NN:
    agent = QNNAgent(env, episodes=episode_count)
else:
    agent = QTableAgent(env, episodes=episode_count)
worker = TrainingWorker(env, agent, episode_count, maximum_steps)
main_app = MainApp(worker)
main_app.start()
