import sys
import serial
from PyQt5 import QtWidgets
from display import Ui_MainWindow

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):	
        super().__init__()
        self.initUI()
        self.ui.calc_tax_button.clicked.connect(self.CalculateTax)

    def initUI(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def CalculateTax(self):
        price = int(self.ui.price_box.toPlainText())
        tax = (self.ui.tax_rate.value())
        total_price = price  + ((tax / 100) * price)
        total_price_string = "The total price with tax is: " + str(total_price)
        self.ui.results_window.setText(total_price_string)	

if __name__ == '__main__' :
    app = QtWidgets.QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
