import train
from q_learning_example.table_env import TableEnv
from q_learning_example.qnn_agent import QNNAgent
from q_learning_example.q_table_agent import QTableAgent

import sys
from PyQt5.QtWidgets import QApplication
import PyQt5.uic as uic

app = QApplication(sys.argv)
screensize = app.desktop().screenGeometry()

main_window = uic.loadUi("./main_window.ui")
main_window.setMinimumSize(screensize.width() // 1.5,
                           screensize.height() // 1.5)

main_window.actionExit.triggered.connect(main_window.close)
main_window.show()

process_event_countdown = 10


def output(text: str) -> None:
    global process_event_countdown
    main_window.plainTextEdit.appendPlainText(text)
    if process_event_countdown == 0:
        QApplication.processEvents()
        process_event_countdown = 10
    process_event_countdown -= 1


env = TableEnv()

USE_NN = True
episode_count = 200
maximum_steps = 600
if USE_NN:
    agent = QNNAgent(env, episodes=episode_count)
else:
    agent = QTableAgent(env, episodes=episode_count)

train.train(env, agent, episode_count, maximum_steps, output, use_cpu=True)
sys.exit(app.exec_())
