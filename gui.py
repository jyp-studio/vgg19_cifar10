# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt6 UI code generator 6.4.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 350)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 10, 261, 281))
        self.groupBox.setObjectName("groupBox")
        self.btnLoad = QtWidgets.QPushButton(self.groupBox)
        self.btnLoad.setGeometry(QtCore.QRect(10, 30, 241, 32))
        self.btnLoad.setObjectName("btnLoad")
        self.btn5_1 = QtWidgets.QPushButton(self.groupBox)
        self.btn5_1.setGeometry(QtCore.QRect(10, 70, 241, 32))
        self.btn5_1.setObjectName("btn5_1")
        self.btn5_2 = QtWidgets.QPushButton(self.groupBox)
        self.btn5_2.setGeometry(QtCore.QRect(10, 110, 241, 32))
        self.btn5_2.setObjectName("btn5_2")
        self.btn5_3 = QtWidgets.QPushButton(self.groupBox)
        self.btn5_3.setGeometry(QtCore.QRect(10, 150, 241, 32))
        self.btn5_3.setObjectName("btn5_3")
        self.btn5_4 = QtWidgets.QPushButton(self.groupBox)
        self.btn5_4.setGeometry(QtCore.QRect(10, 190, 241, 32))
        self.btn5_4.setObjectName("btn5_4")
        self.btn5_5 = QtWidgets.QPushButton(self.groupBox)
        self.btn5_5.setGeometry(QtCore.QRect(10, 230, 241, 32))
        self.btn5_5.setObjectName("btn5_5")
        self.labelImg = QtWidgets.QLabel(self.centralwidget)
        self.labelImg.setGeometry(QtCore.QRect(330, 30, 291, 261))
        self.labelImg.setObjectName("labelImg")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 641, 37))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VGG19"))
        self.groupBox.setTitle(_translate("MainWindow", "5. VGG19"))
        self.btnLoad.setText(_translate("MainWindow", "Load Image"))
        self.btn5_1.setText(_translate("MainWindow", "1. Show Train Image"))
        self.btn5_2.setText(_translate("MainWindow", "2. Show Model Structure"))
        self.btn5_3.setText(_translate("MainWindow", "3. Show Data Augmentation"))
        self.btn5_4.setText(_translate("MainWindow", "4. Show Accuracy and Loss"))
        self.btn5_5.setText(_translate("MainWindow", "5. Inference"))
        self.labelImg.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())