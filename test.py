import ctypes
import math
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar

app = QApplication(sys.argv)
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

mainWindow = QMainWindow()
mainWindow.setWindowTitle('Survival DNN')
mainWindow.setMinimumSize(math.floor(
    screensize[0] * 2 / 3), math.floor(screensize[1] * 2 / 3))
menuBar = QMenuBar(mainWindow)
menuBar.setMinimumWidth(mainWindow.minimumWidth())
optionMenu = menuBar.addMenu('option')
exitButton = optionMenu.addAction('Exit')
exitButton.setShortcut(Qt.Key_Escape)
exitButton.setStatusTip('Exit application')
exitButton.triggered.connect(mainWindow.close)
optionMenu.addAction(exitButton)

mainWindow.show()
sys.exit(app.exec_())
