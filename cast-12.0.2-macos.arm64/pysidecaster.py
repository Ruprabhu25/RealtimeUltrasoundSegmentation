#!/usr/bin/env python

import ctypes
import datetime
import os.path
import sys
from pathlib import Path
from typing import Final
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
# sys.path.append("C:\\Users\\Junfei\\Desktop\\Repos\\RealtimeUltrasoundSegmentation")
#from Efficientunet.efficientunet import get_efficientunet_b0
from skimage.transform import resize
from us_unet2 import MultiHeadUNet, UNet
import cv2
import matplotlib.pyplot as plt

if sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle  # load the libcast.so shared library
    pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")  # load the pyclariuscast.so shared library

import pyclariuscast
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot
import time
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14

# frame_num = 0
# quaternions = pd.DataFrame(columns=['qw', 'qx', 'qy', 'qz'])
# time_run = datetime.datetime.now()
# os.makedirs(f"./images/{time_run}")

device = 'cpu'
model = MultiHeadUNet(heads=3, feat_dim=64, out_ch=1).to(device)
model.load_state_dict(torch.load(os.getenv("MODEL_PATH"), map_location=device))
model.eval()

# custom event for handling change in freeze state
class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen


# custom event for handling button presses
class ButtonEvent(QtCore.QEvent):
    def __init__(self, btn, clicks):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks


# custom event for handling new images
class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))


# manages custom events posted from callbacks, then relays as signals to the main widget
class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.usimage = QtGui.QImage()

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        return True


# global required for the cast api callbacks
signaller = Signaller()


# draws the ultrasound image
class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast):
        QtWidgets.QGraphicsView.__init__(self)
        self.cast = cast
        self.setScene(QtWidgets.QGraphicsScene())

    # set the new image and redraw
    def updateImage(self, img):
        #segmented_img = self.segment_image(img)
        segmented_img = img
        self.image = segmented_img
        self.scene().invalidate()

    # saves a local image
    def saveImage(self):
        self.image.save(str(Path.home() / "Pictures/clarius_image.png"))

    # resize the scan converter, image, and scene
    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        self.cast.setOutputSize(w, h)
        self.image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.black)
        self.setSceneRect(0, 0, w, h)

    # black background
    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    # draws the image
    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)

    def segment_image(self, img):
        '''try:
            global frame_num
            img.save(f"./image/{timestamp}/{frame_num}.png")
            frame_num += 1
        except Exception as e:
            print(e)'''
        img_np = self.qimage_to_numpy(img)
        original_height, original_width = img_np.shape[:2]  # Get the original image size
        
        # Convert to PIL Image (for compatibility with torchvision transforms)
        img_pil = Image.fromarray(img_np)

        # Define the necessary transforms (resize, normalize, etc.)
        transform = transforms.Compose([
            transforms.CenterCrop(218),        # Crop the center 218x218
            transforms.Resize((512, 512)),     # Resize to 512x512
            transforms.ToTensor(),             # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)  # Run the model

        # Post-process the model output
        segmentation_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Get the class with the highest score

        # Resize the segmentation map back to the original image size
        resized_segmentation_map = self.resize_segmentation_map(segmentation_map, (original_width, original_height))

        # Map the output to a color for visualization
        return self.apply_colormap(resized_segmentation_map)

    def resize_segmentation_map(self, segmentation_map, target_size):
        resized_map = resize(segmentation_map, target_size, order=1, mode='reflect', anti_aliasing=True)
        return (resized_map > 0.5).astype(np.uint8)  # Threshold for segmentation map

    def apply_colormap(self, segmentation_map):
        colormap = np.array([
            [0, 0, 0],      # background
            [255, 0, 0],    # (red)
        ])
        
        # Map segmentation output to colors
        segmented_image = colormap[segmentation_map]
        return segmented_image

    def convert_to_qimage(self, img_np):
        """
        Convert a NumPy array (image) into a QImage.
        """
        height, width, channels = img_np.shape
        bytes_per_line = channels * width
        qimage = QtGui.QImage(img_np.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        return qimage

    def qimage_to_numpy(self, image: QtGui.QImage):
        """
        Convert a QImage to a NumPy array.
        """
        image = image.convertToFormat(QtGui.QImage.Format_Grayscale8)
        width = image.width()
        height = image.height()

        ptr = image.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=height * image.bytesPerLine())
        arr = arr.reshape((height, image.bytesPerLine()))
        return arr[:, :width]


# main widget with controls and ui
class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.cast = cast
        self.setWindowTitle("Clarius Cast Demo")

        # create central widget within main window
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        ip = QtWidgets.QLineEdit("192.168.1.1")
        ip.setInputMask("000.000.000.000")
        port = QtWidgets.QLineEdit("36035")
        port.setInputMask("00000")

        conn = QtWidgets.QPushButton("Connect")
        self.run = QtWidgets.QPushButton("Run")
        quit = QtWidgets.QPushButton("Quit")
        depthUp = QtWidgets.QPushButton("< Depth")
        depthDown = QtWidgets.QPushButton("> Depth")
        gainInc = QtWidgets.QPushButton("> Gain")
        gainDec = QtWidgets.QPushButton("< Gain")
        captureImage = QtWidgets.QPushButton("Capture Image")
        captureCine = QtWidgets.QPushButton("Capture Movie")
        saveImage = QtWidgets.QPushButton("Save Local")
        bMode = QtWidgets.QPushButton("B Mode")
        cfiMode = QtWidgets.QPushButton("Color Mode")

        # try to connect/disconnect to/from the probe
        def tryConnect():
            try: 
                global frame_num
                frame_num = 0
            except Exception as e:
                print(e)
            if not cast.isConnected():
                if cast.connect(ip.text(), int(port.text()), "research"):
                    self.statusBar().showMessage("Connected")
                    conn.setText("Disconnect")
                else:
                    self.statusBar().showMessage("Failed to connect to {0}".format(ip.text()))
            else:
                if cast.disconnect():
                    self.statusBar().showMessage("Disconnected")
                    conn.setText("Connect")
                else:
                    self.statusBar().showMessage("Failed to disconnect")

        # try to freeze/unfreeze
        def tryFreeze():
            if cast.isConnected():
                cast.userFunction(CMD_FREEZE, 0)

        # try depth up
        def tryDepthUp():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_DEC, 0)

        # try depth down
        def tryDepthDown():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_INC, 0)

        # try gain down
        def tryGainDec():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_DEC, 0)

        # try gain up
        def tryGainInc():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_INC, 0)

        # try capture image
        def tryCaptureImage():
            if cast.isConnected():
                cast.userFunction(CMD_CAPTURE_IMAGE, 0)

        # try capture cine
        def tryCaptureCine():
            if cast.isConnected():
                cast.userFunction(CMD_CAPTURE_CINE, 0)

        # try to save a local image
        def trySaveImage():
            self.img.saveImage()

        # try b mode
        def tryBMode():
            if cast.isConnected():
                cast.userFunction(CMD_B_MODE, 0)

        # try cfi mode
        def tryCfiMode():
            if cast.isConnected():
                cast.userFunction(CMD_CFI_MODE, 0)

        conn.clicked.connect(tryConnect)
        self.run.clicked.connect(tryFreeze)
        quit.clicked.connect(self.shutdown)
        depthUp.clicked.connect(tryDepthUp)
        depthDown.clicked.connect(tryDepthDown)
        gainInc.clicked.connect(tryGainInc)
        gainDec.clicked.connect(tryGainDec)
        captureImage.clicked.connect(tryCaptureImage)
        captureCine.clicked.connect(tryCaptureCine)
        saveImage.clicked.connect(trySaveImage)
        bMode.clicked.connect(tryBMode)
        cfiMode.clicked.connect(tryCfiMode)

        # add widgets to layout
        self.img = ImageView(cast)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img)

        inplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(inplayout)
        inplayout.addWidget(ip)
        inplayout.addWidget(port)

        connlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(connlayout)
        connlayout.addWidget(conn)
        connlayout.addWidget(self.run)
        connlayout.addWidget(quit)
        central.setLayout(layout)

        prmlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(prmlayout)
        prmlayout.addWidget(depthUp)
        prmlayout.addWidget(depthDown)
        prmlayout.addWidget(gainDec)
        prmlayout.addWidget(gainInc)

        caplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(caplayout)
        caplayout.addWidget(captureImage)
        caplayout.addWidget(captureCine)
        caplayout.addWidget(saveImage)

        modelayout = QtWidgets.QHBoxLayout()
        layout.addLayout(modelayout)
        modelayout.addWidget(bMode)
        modelayout.addWidget(cfiMode)

        # connect signals
        signaller.freeze.connect(self.freeze)
        signaller.button.connect(self.button)
        signaller.image.connect(self.image)

        # get home path
        path = os.path.expanduser("~/")
        if cast.init(path, 640, 480):
            self.statusBar().showMessage("Initialized")
        else:
            self.statusBar().showMessage("Failed to initialize")

    # handles freeze messages
    @Slot(bool)
    def freeze(self, frozen):
        if frozen:
            self.run.setText("Run")
            self.statusBar().showMessage("Image Stopped")
        else:
            self.run.setText("Freeze")
            self.statusBar().showMessage("Image Running (check firewall settings if no image seen)")

    # handles button messages
    @Slot(int, int)
    def button(self, btn, clicks):
        self.statusBar().showMessage("Button {0} pressed w/ {1} clicks".format(btn, clicks))

    # handles new images
    @Slot(QtGui.QImage)
    def image(self, img):
        self.img.updateImage(img)

    # handles shutdown
    @Slot()
    def shutdown(self):
        if sys.platform.startswith("linux"):
            # unload the shared library before destroying the cast object
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
        self.cast.destroy()
        quaternions.to_csv(f"./positions/quaternion_run_{time_run}.csv")
        QtWidgets.QApplication.quit()


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
    image_size = (128, 128)

    if bpp == 4:
        img_qt = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
        img_pil = Image.frombytes("RGBA", (width, height), image)
        #print("rgb")
    else:
        img_qt = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
        img_pil = Image.frombytes("L", (width, height), image)
        #print("grayscale")

    # Resize image using OpenCV
    try:
        img_np = np.array(img_pil)
        img_resized = cv2.resize(img_np, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
        img_resized = img_resized[:, :, :3]

    # Apply weights to average RGB
        img_resized = np.dot(img_resized, [0.299, 0.587, 0.114])

        img_resized = img_resized.reshape(128, 128, 1)
        #print(img_resized.shape)
        img_resized = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0)
        img_resized = img_resized.permute(0, 3, 1, 2)
        #print(img_resized.shape)

        # 6/12/25 try to use model
        with torch.no_grad():
            _, pred = model(img_resized)
            print(pred.shape)
            pred = pred.squeeze(0).numpy()
        # Normalize and convert to torch
        #img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0)

        pred_img = Image.fromarray(pred.reshape(128, 128))
        pred_img.save(f"./cast-12.0.2-macos.arm64/images/test.png")
        # Convert resized image back to QImage
        img_qt_resized = QtGui.QImage(
            pred.astype(np.uint8).copy(), 
            image_size[1], 
            image_size[0], 
            image_size[1],  # bytesPerLine for grayscale: width * 1
            QtGui.QImage.Format_Grayscale8
        )
    except Exception as e:
        print(e)
        img_qt_resized = img_qt

    signaller.usimage = img_qt_resized  # Clone to detach from NumPy memory
    evt = ImageEvent()
    QtCore.QCoreApplication.postEvent(signaller, evt)

    # print(imu[0].qw, imu[0].qx, imu[0].qy, imu[0].qz)
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


## main function
def main():
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, newImuData, freezeFn, buttonsFn)
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWidget(cast)
    widget.resize(640, 480)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
