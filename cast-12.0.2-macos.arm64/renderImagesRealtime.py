import sys
import os
import ctypes
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot
from PIL import Image

# If on Linux, load native shared libraries
if sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle
    pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")
else:
    import pyclariuscast

from efficientunet import get_efficientunet_b0

# Constants
CMD_FREEZE = 1
CMD_CAPTURE_IMAGE = 2
CMD_CAPTURE_CINE = 3
CMD_DEPTH_DEC = 4
CMD_DEPTH_INC = 5
CMD_GAIN_DEC = 6
CMD_GAIN_INC = 7
CMD_B_MODE = 12
CMD_CFI_MODE = 14

# Global state
frame_num = 0
time_run = datetime.datetime.now()

# ----------------------------- Real-Time Plotting ----------------------------- #
class RealTime3DPlotter:
    def __init__(self, image_size=5, distance=10):
        self.image_size = image_size
        self.distance = distance
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-15, 15])
        self.ax.set_ylim([-15, 15])
        self.ax.set_zlim([-15, 15])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.ion()
        plt.show()

    def quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

    def add_image(self, image_np, quaternion):
        R = self.quaternion_to_rotation_matrix(quaternion)
        normal = R[:, 2]
        center = self.distance * normal

        u = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(normal, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)

        u *= self.image_size
        v *= self.image_size

        h, w = image_np.shape[:2]
        corners = np.array([
            center - u - v,
            center + u - v,
            center + u + v,
            center - u + v
        ])

        x = np.linspace(corners[0, 0], corners[2, 0], w)
        y = np.linspace(corners[0, 1], corners[2, 1], h)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, corners[0, 2])

        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32) / 255.0

        self.ax.plot_surface(X, Y, Z, rstride=4, cstride=4, facecolors=image_np, shade=False)
        plt.draw()
        plt.pause(0.001)

def qimage_to_numpy_array(qimage):
    qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.array(ptr).reshape((height, width, 3))
    return arr

# Global plotter instance
real_time_plotter = RealTime3DPlotter()

# ----------------------------- Signal Handler ----------------------------- #
class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen

class ButtonEvent(QtCore.QEvent):
    def __init__(self, btn, clicks):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks

class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))

class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.usimage = QtGui.QImage()

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        return True

signaller = Signaller()

# ----------------------------- Image Display ----------------------------- #
class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast):
        super().__init__()
        self.cast = cast
        self.setScene(QtWidgets.QGraphicsScene())
        self.image = QtGui.QImage()

    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()

    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        self.cast.setOutputSize(w, h)
        self.image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        self.image.fill(Qt.black)
        self.setSceneRect(0, 0, w, h)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, Qt.black)

    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)

# ----------------------------- Main Window ----------------------------- #
class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast):
        super().__init__()
        self.cast = cast
        self.setWindowTitle("Clarius Cast Live Visualizer")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.img = ImageView(cast)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img)

        conn = QtWidgets.QPushButton("Connect")
        run = QtWidgets.QPushButton("Run")
        quit = QtWidgets.QPushButton("Quit")
        ip = QtWidgets.QLineEdit("192.168.1.1")
        ip.setInputMask("000.000.000.000")
        port = QtWidgets.QLineEdit("5828")
        port.setInputMask("00000")

        conn.clicked.connect(lambda: self.tryConnect(ip.text(), port.text(), conn))
        run.clicked.connect(self.tryFreeze)
        quit.clicked.connect(self.shutdown)

        connlayout = QtWidgets.QHBoxLayout()
        connlayout.addWidget(ip)
        connlayout.addWidget(port)
        connlayout.addWidget(conn)
        connlayout.addWidget(run)
        connlayout.addWidget(quit)

        layout.addLayout(connlayout)
        central.setLayout(layout)

        signaller.freeze.connect(self.freeze)
        signaller.button.connect(self.button)
        signaller.image.connect(self.image)

        path = os.path.expanduser("~/")
        if cast.init(path, 640, 480):
            self.statusBar().showMessage("Initialized")
        else:
            self.statusBar().showMessage("Failed to initialize")

    def tryConnect(self, ip, port, button):
        if not self.cast.isConnected():
            if self.cast.connect(ip, int(port), "research"):
                self.statusBar().showMessage("Connected")
                button.setText("Disconnect")
            else:
                self.statusBar().showMessage("Failed to connect")
        else:
            if self.cast.disconnect():
                self.statusBar().showMessage("Disconnected")
                button.setText("Connect")
            else:
                self.statusBar().showMessage("Failed to disconnect")

    def tryFreeze(self):
        if self.cast.isConnected():
            self.cast.userFunction(CMD_FREEZE, 0)

    @Slot(bool)
    def freeze(self, frozen):
        status = "Image Stopped" if frozen else "Image Running"
        self.statusBar().showMessage(status)

    @Slot(int, int)
    def button(self, btn, clicks):
        self.statusBar().showMessage(f"Button {btn} pressed {clicks} times")

    @Slot(QtGui.QImage)
    def image(self, img):
        self.img.updateImage(img)

    def shutdown(self):
        self.cast.shutdown()
        QtCore.QCoreApplication.quit()


## called when a new processed image is streamed
# @param image the scan-converted image data
# @param width width of the image in pixels
# @param height height of the image in pixels
# @param sz full size of image
# @param micronsPerPixel microns per pixel
# @param timestamp the image timestamp in nanoseconds
# @param angle acquisition angle for volumetric data
# @param imu inertial data tagged with the frame
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    bpp = sz / (width * height)
    if bpp == 4:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    else:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)

    if bpp == 4:
        img_graph = Image.frombytes("RGBA", (width, height), image)
    else:
        img_graph = Image.frombytes("L", (width, height), image)
    img_graph_np = np.array(img_graph)
    real_time_plotter.add_image(img_graph_np, (imu[0].qw, imu[0].qx, imu[0].qy, imu[0].qz))
    # a deep copy is important here, as the memory from 'image' won't be valid after the event posting
    signaller.usimage = img.copy()
    evt = ImageEvent()
    QtCore.QCoreApplication.postEvent(signaller, evt)
    # try:
    #     global quaternions
    #     global time_run
    #     global frame_num
    #     new_row = pd.DataFrame([
    #         {'qw': imu[0].qw, 'qx': imu[0].qx, 'qy': imu[0].qy, 'qz': imu[0].qz}
    #     ])
    #     quaternions = pd.concat(
    #         [quaternions, 
    #         new_row]
    #     )
    #     print(f"saving {frame_num}")
    #     img_save.save(f"./images/{time_run}/{frame_num}.png")
    #     print(f"saved {frame_num}")
    #     frame_num += 1
    # except Exception as e:
    #     print(e)
    return


## called when a new raw image is streamed
# @param image the raw pre scan-converted image data, uncompressed 8-bit or jpeg compressed
# @param lines number of lines in the data
# @param samples number of samples in the data
# @param bps bits per sample
# @param axial microns per sample
# @param lateral microns per line
# @param timestamp the image timestamp in nanoseconds
# @param jpg jpeg compression size if the data is in jpeg format
# @param rf flag for if the image received is radiofrequency data
# @param angle acquisition angle for volumetric data
def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    return


## called when a new spectrum image is streamed
# @param image the spectral image
# @param lines number of lines in the spectrum
# @param samples number of samples per line
# @param bps bits per sample
# @param period line repetition period of spectrum
# @param micronsPerSample microns per sample for an m spectrum
# @param velocityPerSample velocity per sample for a pw spectrum
# @param pw flag that is true for a pw spectrum, false for an m spectrum
def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


## called when a new imu data is streamed
# @param imu inertial data tagged with the frame
def newImuData(imu):
    return


## called when freeze state changes
# @param frozen the freeze state
def freezeFn(frozen):
    evt = FreezeEvent(frozen)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


## called when a button is pressed
# @param button the button that was pressed
# @param clicks number of clicks performed
def buttonsFn(button, clicks):
    evt = ButtonEvent(button, clicks)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return

# ----------------------------- Setup and Main Loop ----------------------------- #
if __name__ == "__main__":
    # model = get_efficientunet_b0(out_channels=1)
    # model.load_state_dict(torch.load("cast-12.0.2-macos.arm64/EfficientUNet.pth", map_location=torch.device("cpu")))
    # model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    app = QtWidgets.QApplication(sys.argv)
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, newImuData, freezeFn, buttonsFn)
    win = MainWidget(cast)
    win.resize(720, 640)
    win.show()
    sys.exit(app.exec())
