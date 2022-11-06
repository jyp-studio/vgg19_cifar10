from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from gui import Ui_MainWindow

from PIL import Image
from data import *
import traceback, sys
import cv2


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class Controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        self.device = "mps"
        self.data = Data()
        self.filename = ""

        # get some random training images
        dataiter = iter(
            torch.utils.data.DataLoader(
                self.data.trainset, batch_size=9, shuffle=True, num_workers=0
            )
        )
        self.images, self.labels = next(dataiter)

        self.threadpool = QThreadPool()

    def setup_control(self):
        self.ui.btnLoad.clicked.connect(self.open_file)
        self.ui.btn5_1.clicked.connect(self.show5_1)
        self.ui.btn5_2.clicked.connect(self.show5_2)
        self.ui.btn5_3.clicked.connect(self.show5_3)
        self.ui.btn5_4.clicked.connect(self.show5_4)
        self.ui.btn5_5.clicked.connect(self.show5_5)

    def open_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.display_img()

    def display_img(self):
        self.img = cv2.imread(self.filename)
        # self.img = Image.open(self.filename)
        # tensor = self.data.transform(self.img)
        # self.img = self.data.unnormalize(tensor)
        self.img = cv2.resize(self.img, (224, 224), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QtGui.QImage(
            self.img, width, height, bytesPerline, QtGui.QImage.Format.Format_RGB888
        ).rgbSwapped()
        self.ui.labelImg.setPixmap(QtGui.QPixmap.fromImage(self.qimg))

    def show5_1(self):
        # show images
        plt.figure()
        for index, img in enumerate(self.images, start=1):
            npimg = self.data.unnormalize(img)
            plt.subplot(3, 3, index)
            plt.title(f"{self.data.classes[labels[index - 1]]:5s}", fontsize=10)
            plt.imshow(npimg)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def show5_2(self):
        self.net = VGG19(10)
        summary(self.net, (3, 32, 32), device="cpu")
        self.net.to(device=self.device)

    def show5_3(self):
        plt.figure()
        origin = self.data.unnormalize(self.images[0])
        plt.subplot(2, 3, 2)
        plt.title("origin")
        plt.imshow(origin)
        plt.xticks([])
        plt.yticks([])
        for i in range(4, 7):
            tensor = self.data.augmentation(self.images[0])
            trans = self.data.unnormalize(tensor)
            plt.subplot(2, 3, i)
            plt.title(f"trans{i-3}")
            plt.imshow(trans)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def show5_4(self):
        pass

    def show5_5(self):
        pass
