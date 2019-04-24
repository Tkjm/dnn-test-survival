import train
from q_learning_example.table_env import TableEnv
from q_learning_example.qnn_agent import QNNAgent
from q_learning_example.q_table_agent import QTableAgent
from main_window import MainWindow

import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
main_window = MainWindow(app)
env = TableEnv()

USE_NN = True
episode_count = 200
maximum_steps = 600
if USE_NN:
    agent = QNNAgent(env, episodes=episode_count)
else:
    agent = QTableAgent(env, episodes=episode_count)

train.train(env, agent, episode_count, maximum_steps,
            main_window.output, use_cpu=True)
sys.exit(app.exec_())
