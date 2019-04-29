from main_window import MainWindow

from PyQt5.QtWidgets import QApplication
import sys
from PyQt5.QtCore import QThread, QObject


class MainApp():
    def __init__(self, worker: QObject) -> None:
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication([])
        self.main_window = MainWindow()
        self.worker = worker

    def start(self) -> None:
        thread = QThread()
        thread.started.connect(self.worker.start)
        self.worker.moveToThread(thread)
        self.worker.output_ready.connect(self.main_window.output)
        self.worker.done.connect(thread.quit)
        thread.finished.connect(self.worker.deleteLater)
        thread.start()
        exit_code = self.app.exec_()
        self.worker.stop()
        thread.quit()
        thread.wait()
        self.main_window.deleteLater()
        sys.exit(exit_code)
