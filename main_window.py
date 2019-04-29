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
        self.output_group = {}
        self.ui.listWidget.currentItemChanged.connect(self.show_current_output)
        self.ui.show()

    @pyqtSlot(str, str)
    def output(self, group: str, text: str) -> None:
        if group not in self.output_group:
            self.output_group[group] = text
            self.ui.listWidget.addItem(group)
        else:
            self.output_group[group] += "\n" + text
            current_item = self.ui.listWidget.currentItem()
            if current_item is not None and current_item.text() == group:
                self.ui.plainTextEdit.appendPlainText(text)

    @pyqtSlot()
    def show_current_output(self) -> None:
        self.ui.plainTextEdit.setPlainText(
            self.output_group[self.ui.listWidget.currentItem().text()])
