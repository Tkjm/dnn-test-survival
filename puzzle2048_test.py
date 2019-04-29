from puzzle2048.puzzle2048_env import Puzzle2048_env
from puzzle2048.qnn_agent import QNNAgent
from main_app import MainApp

from training_worker import TrainingWorker
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
env = Puzzle2048_env()
episode_count = 200
maximum_steps = 600
agent = QNNAgent(env, episodes=episode_count)
worker = TrainingWorker(env, agent, episode_count, maximum_steps)
main_app = MainApp(worker)
main_app.start()
