from PyQt5.QtWidgets import QApplication
import PyQt5.uic as uic
import time


class MainWindow():
    def __init__(self, app: QApplication) -> None:
        self.ui = uic.loadUi("./main_window.ui")
        screensize = app.desktop().screenGeometry()
        self.ui.setMinimumSize(screensize.width() // 1.5,
                               screensize.height() // 1.5)
        self.ui.actionExit.triggered.connect(self.ui.close)
        self.ui.show()
        self.process_event_countdown = 10
        self.last_process_time = time.time()

    def output(self, text: str) -> None:
        self.ui.plainTextEdit.appendPlainText(text)
        if self.process_event_countdown == 0\
                or time.time() - self.last_process_time > 0.1:
            QApplication.processEvents()
            self.last_process_time = time.time()
            self.process_event_countdown = 10
        self.process_event_countdown -= 1
