from PyQt5.QtWidgets import QApplication
import PyQt5.uic as uic
from PyQt5.QtCore import pyqtSlot, QObject


class MainWindow(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.ui = uic.loadUi("./main_window.ui")
        screen_size = QApplication.desktop().screenGeometry()
        self.ui.setMinimumSize(screen_size.width() // 1.5,
                               screen_size.height() // 1.5)
        self.ui.show()

    @pyqtSlot(str)
    def output(self, text: str) -> None:
        self.ui.plainTextEdit.appendPlainText(text)
