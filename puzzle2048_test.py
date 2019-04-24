import train
from puzzle2048.puzzle2048_env import Puzzle2048_env
from puzzle2048.qnn_agent import QNNAgent
from main_window import MainWindow

import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
main_window = MainWindow(app)
env = Puzzle2048_env()

episode_count = 200
maximum_steps = 600
agent = QNNAgent(env, episodes=episode_count)

train.train(env, agent, episode_count, maximum_steps,
            main_window.output, use_cpu=True)
sys.exit(app.exec_())
