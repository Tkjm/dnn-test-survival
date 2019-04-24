import q_learning_example.test as test

import sys
from PyQt5.QtWidgets import QApplication
import PyQt5.uic as uic

app = QApplication(sys.argv)
screensize = app.desktop().screenGeometry()

mainWindow = uic.loadUi("./main_window.ui")
mainWindow.setMinimumSize(screensize.width() // 1.5,
                          screensize.height() // 1.5)

mainWindow.actionExit.triggered.connect(mainWindow.close)
mainWindow.show()


def output(text: str) -> None:
    mainWindow.plainTextEdit.appendPlainText(text)
    QApplication.processEvents()


test.train(output, use_cpu=True)
sys.exit(app.exec_())
