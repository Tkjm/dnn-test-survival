import sys
from PyQt5.QtWidgets import QApplication, QLabel
app = QApplication([])
label = QLabel('Hello DNN!')
label.show()
sys.exit(app.exec_())
