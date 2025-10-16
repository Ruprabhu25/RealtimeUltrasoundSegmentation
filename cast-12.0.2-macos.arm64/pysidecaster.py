#!/usr/bin/env python

import argparse
import ctypes
import datetime
import os.path
import sys
from pathlib import Path
from typing import Final
from PIL import Image
import torch
import numpy as np
from scipy.interpolate import splprep, splev
from model.us_unet2 import MultiHeadUNet
import cv2
import matplotlib.pyplot as plt
import pyclariuscast
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Slot
import pandas as pd
import os
from dotenv import load_dotenv
from matplotlib.colors import LightSource
from convex_hull import (
    extract_3d_hull_from_image,
    get_rotation_center,
    update_plot_with_new_frame,
    quaternion_distance,
    MIN_ROTATION_RAD,
    MAX_ROTATION_RAD,
    add_shaded_quad,
)
from dataclasses import dataclass
from logging import getLogger

# sys.path.append("C:\\Users\\Junfei\\Desktop\\Repos\\RealtimeUltrasoundSegmentation")
# from Efficientunet.efficientunet import get_efficientunet_b0

CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14

from PySide6.QtCore import QObject, Signal

class PlotSignaller(QObject):
    plot_update = Signal(object, object)  # hull_3d, all_hulls_3d

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
        self.image = img
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


# main widget with controls and ui
class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.cast = cast
        self.setWindowTitle("MHU Realtime Ultrasound Segmentation")

        # create central widget within main window
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        ip = QtWidgets.QLineEdit("192.168.1.1")
        ip.setInputMask("000.000.000.000")
        port = QtWidgets.QLineEdit("5828")
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
        segmentBtn = QtWidgets.QCheckBox("Segment Image")
        plotBtn = QtWidgets.QCheckBox("Plot")
        modelFile = QtWidgets.QLineEdit(text="best_mhu-2.pth", placeholderText="Model File name")
        saveResults = QtWidgets.QCheckBox("Save Results")

        # try to connect/disconnect to/from the probe
        def tryConnect():
            if not cast.isConnected():
                if cast.connect(ip.text(), int(port.text()), "research"):
                    self.statusBar().showMessage("Connected")
                    conn.setText("Disconnect")
                    try:
                        seg_plot.frame_num = 0
                        seg_plot.save_results = saveResults.isChecked()
                        print(f"saveresults is checked? {seg_plot.save_results} {saveResults.isChecked()}")
                        seg_plot.model_file = modelFile.text()
                        seg_plot.initialize()
                    except Exception as e:
                        print(e)
                else:
                    self.statusBar().showMessage(
                        "Failed to connect to {0}".format(ip.text())
                    )
            else:
                print("trying to disconnect")
                if cast.disconnect():
                    print("disconnected")
                    if seg_plot.save_results:
                        print(f"size of quaternion data: seg_plot.quaternions.count()")
                        print("saving quaternion data")
                        # seg_plot.quaternions.to_csv(
                        #     f"{seg_plot.positions_path}/quaternion_run_{seg_plot.time_run}.csv",
                        #     columns=["qw", "qx", "qy", "qz"],
                        #     index=False,
                        # )
                        seg_plot.positions_fp.close()
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

        def trySegmentImage():
            if seg_plot.segment_image:
                seg_plot.segment_image = False
                self.statusBar().showMessage("Stopped segmenting image")
            else:
                seg_plot.segment_image = True
                self.statusBar().showMessage("Segmenting image...")

        def tryPlot():
            if seg_plot.plot:
                seg_plot.plot = False
                self.statusBar().showMessage("Stopped Plotting")
            else:
                seg_plot.segment_image = True
                seg_plot.plot = True
                self.statusBar().showMessage("Plotting...")

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
        segmentBtn.clicked.connect(trySegmentImage)
        plotBtn.clicked.connect(tryPlot)

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

        # Add new buttons to layout
        layout.addWidget(modelFile)
        layout.addWidget(saveResults)
        layout.addWidget(segmentBtn)
        layout.addWidget(plotBtn)

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
            self.statusBar().showMessage(
                "Image Running (check firewall settings if no image seen)"
            )

    # handles button messages
    @Slot(int, int)
    def button(self, btn, clicks):
        self.statusBar().showMessage(
            "Button {0} pressed w/ {1} clicks".format(btn, clicks)
        )

    # handles new images
    @Slot(QtGui.QImage)
    def image(self, img):
        self.img.updateImage(img)

    # handles shutdown
    @Slot()
    def shutdown(self):
        print("trying to shutdown")
        if sys.platform.startswith("linux"):
            # unload the shared library before destroying the cast object
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
        self.cast.destroy()
        print("trying to shutdown")
        # if seg_plot.save_results:
        #     print("saving quaternion data")
        #     seg_plot.quaternions.to_csv(
        #         f"{seg_plot.positions_path}/quaternion_run_{seg_plot.time_run}.csv",
        #         columns=["qw", "qx", "qy", "qz"],
        #         index=False,
        #     )
        seg_plot.positions_fp.close()
        print("Shutting down plot")
        if seg_plot.plot_initialized:
            plt.ioff()
            plt.close("all")
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
    seg_plot.frame_num += 1
    print(f"ingested {seg_plot.frame_num} at {datetime.datetime.now()}")
    if seg_plot.frame_num % 2 == 0:
        return
    if bpp == 4:
        img_qt = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
        img_pil = Image.frombytes("RGBA", (width, height), image)
    else:
        img_qt = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
        img_pil = Image.frombytes("L", (width, height), image)
    try:
        if seg_plot.segment_image:
            pred, pred_img = segment_image(img_pil, image_size)
            pred_img_resized = pred_img.resize((width, height))
            img_qt_resized, largest_contour = display_segmented_image(img_pil, image_size, pred, width, height)
        else:
            img_qt_resized = img_qt
        
        if seg_plot.save_results:
                seg_plot.positions_fp.write(f"{imu[0].qw},{imu[0].qx},{imu[0].qy},{imu[0].qz}\n")
                print(f"saving {seg_plot.frame_num}")
                if seg_plot.segment_image and largest_contour is not None:
                    #pred_img_resized.save(f"{seg_plot.images_path}/{seg_plot.time_run}/{seg_plot.frame_num}_segmented.png")
                    pts = largest_contour.reshape(-1, 2)
                    np.savez(f"{seg_plot.images_path}/{seg_plot.time_run}/{seg_plot.frame_num}_contour.png", pts=pts)
                img_pil.save(f"{seg_plot.images_path}/{seg_plot.time_run}/{seg_plot.frame_num}.png")
                print(f"saved {seg_plot.frame_num}")

        if seg_plot.plot:
            plot_frame(imu, pred_img_resized)
    except Exception as e:
        print(e)
        img_qt_resized = img_qt

    signaller.usimage = img_qt_resized
    evt = ImageEvent()
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return

def plot_frame(imu, pred_img):
    quat = [imu[0].qx, imu[0].qy, imu[0].qz, imu[0].qw]
    # Skip if quaternion hasn't changed significantly
    if seg_plot.last_quat is not None:
        angle = quaternion_distance(quat, seg_plot.last_quat)
        if angle < MIN_ROTATION_RAD:
            print(f"Skipped frame due to min threshold: Δangle={np.degrees(angle):.2f}°")
            return
        if angle > MAX_ROTATION_RAD:
            print(f"Skipped frame due to max threshold: Δangle={np.degrees(angle):.2f}°")
            return
    if seg_plot.frame_num % 5 == 0:
        rot, center = get_rotation_center(quat)

        hull_3d = extract_3d_hull_from_image(
            np.array(pred_img.convert("L")), rot, center
        )

        # Update last_quat
        seg_plot.last_quat = quat
        # Emit signal to main thread for plotting
        seg_plot.plot_signaller.plot_update.emit(hull_3d, seg_plot.all_hulls_3d)

def segment_image(img_pil, image_size):
    img_np = np.array(img_pil)
    img_resized = cv2.resize(
        img_np, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA
    )
    img_resized = img_resized[:, :, :3]

    img_resized = np.dot(img_resized, [0.299, 0.587, 0.114])

    img_resized = img_resized.reshape(128, 128, 1)
    img_resized = torch.from_numpy(img_resized.astype(np.float32)/255.0).unsqueeze(0)
    img_resized = img_resized.permute(0, 3, 1, 2)
    print(f"before segmentation {seg_plot.frame_num} at {datetime.datetime.now()}")
    with torch.no_grad():
        _, pred = seg_plot.model(img_resized)
        pred = pred.squeeze(0).numpy()
    print(f"after segmentation {seg_plot.frame_num} at {datetime.datetime.now()}")


    return pred, Image.fromarray((pred.squeeze() * 255).astype(np.uint8))

def smooth_mask_morph(mask, radius=4, cycles=1, order="open-close"):
    if mask.dtype == np.uint8:
        m = np.ascontiguousarray(mask)
    else:
        m = (mask > 0.5).astype(np.uint8, copy=False) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

    if order == "open-close":
        out = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=cycles)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=cycles)
    else:
        out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=cycles)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  k, iterations=cycles)
    return out

def display_segmented_image(img_pil: Image, image_size, pred: np.ndarray, original_width: int, original_height: int):
    pred = pred.squeeze(0)
    seg_mask = (pred * 255).astype(np.uint8)

    # Create blue overlay with transparency
    blue_overlay = np.zeros((128, 128, 4), dtype=np.uint8)
    blue_overlay[..., 2] = 255  # blue
    blue_overlay[..., 3] = (seg_mask * 0.3).astype(np.uint8)  # semi-transparent

    # Convert original image
    original_img = img_pil.convert("RGBA").resize((original_width, original_height))

    # Apply ultrasound border mask
    border_mask_np = np.array(
        seg_plot.ultrasound_border_mask.resize((128, 128), Image.Resampling.BILINEAR)
    )
    white_pixels = np.all(border_mask_np[..., :3] == 255, axis=-1)
    blue_overlay[~white_pixels] = [0, 0, 0, 0]

    # ------- Contour detection (on small prediction) -------
    pred_smoothed = smooth_mask_morph(pred)

    print(f"before finding contours {seg_plot.frame_num} at {datetime.datetime.now()}")
    contours, _ = cv2.findContours(pred_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"after finding contours {seg_plot.frame_num} at {datetime.datetime.now()}")
    print("number of contours found: ", len(contours))

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
    if not valid_contours:
        return None, None

    largest_contour = max(valid_contours, key=cv2.contourArea)

    # -------- Resize overlay first --------
    blue_overlay_img = Image.fromarray(blue_overlay, mode="RGBA").resize((original_width, original_height))
    blue_overlay_np = np.array(blue_overlay_img)

    # -------- Scale and draw contour on resized overlay --------
    scale_x = original_width / 128
    scale_y = original_height / 128
    scaled_contour = np.array([[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in largest_contour])

    # Draw scaled contour in yellow
    print(f"before drawing contours {seg_plot.frame_num} at {datetime.datetime.now()}")
    cv2.drawContours(blue_overlay_np, [scaled_contour], -1, (255, 255, 0, 255), 2)  # RGBA: yellow with full alpha
    print(f"after drawing contours {seg_plot.frame_num} at {datetime.datetime.now()}")

    # Re-convert back to PIL
    blue_overlay_img = Image.fromarray(blue_overlay_np, mode="RGBA")

    # Final compositing
    combined = Image.alpha_composite(original_img, blue_overlay_img)

    # Side-by-side comparison
    final_display = Image.new("RGBA", (original_width * 2, original_height))
    final_display.paste(combined, (0, 0))
    final_display.paste(original_img, (original_width, 0))

    # Return Qt QImage and contour
    return QtGui.QImage(
        final_display.tobytes("raw", "RGBA"),
        final_display.size[0],
        final_display.size[1],
        final_display.size[0] * 4,
        QtGui.QImage.Format_RGBA8888,
    ), largest_contour

def plot_update_handler(hull_3d, all_hulls_3d):
    if not seg_plot.plot_initialized:
        plt.ion()
        seg_plot.fig = plt.figure(figsize=(10, 10))
        seg_plot.ax = seg_plot.fig.add_subplot(111, projection="3d")
        seg_plot.ax.set_box_aspect([1, 1, 1])
        seg_plot.ax.set_xlim(-0.25, 0.25)
        seg_plot.ax.set_ylim(-0.25, 0.25)
        seg_plot.ax.set_zlim(-0.25, 0.25)
        seg_plot.ax.set_xlabel("X-axis")
        seg_plot.ax.set_ylabel("Y-axis")
        seg_plot.ax.set_zlabel("Z-axis")
        seg_plot.plot_initialized = True

    update_plot_with_new_frame(seg_plot.ax, hull_3d, all_hulls_3d)
    plt.draw()

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
def newSpectrumImage(
    image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw
):
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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_colormap(matrix, cmap_name="viridis"):
    """
    Convert a float matrix (H x W) to an RGB image using a matplotlib colormap.
    """
    # Normalize values between 0 and 1
    normed = (matrix - np.min(matrix)) / (np.ptp(matrix) + 1e-8)

    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Apply the colormap (returns RGBA)
    colored = cmap(normed)  # shape: (H, W, 4)

    # Convert to uint8 RGB
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(rgb_image)

# # Example usage
# matrix = np.random.rand(128, 128)  # your float matrix
# img_colored = apply_colormap(matrix, cmap_name="plasma")
# img_colored.show()


@dataclass
class SegmentationPlot:
    device: str = "cpu"  # device to run the model on
    save_results: bool = False  # whether to save results
    log_level: str = "DEBUG"  # logging level
    plot: bool = False # whether to plot image or not
    segment_image: bool = False  # whether to segment the image
    base_dir: str = os.getcwd()  # base directory for saving results
    model_file: str = "best_mhu-2.pth"  # model file name
    plot_signaller: PlotSignaller = None  # signaller for plot updates
    last_quat: list = None
    frame_num: int = 0

    def initialize(self):
        # if saving results, create directories for images and positions
        print("intialize called")
        self.images_path = os.path.join(self.base_dir, "images")
        if self.save_results:
            self.frame_num = 0
            self.time_run = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            os.makedirs(os.path.join(self.images_path, self.time_run), exist_ok=True)
            self.positions_path = os.path.join(self.base_dir, "positions")
            os.makedirs(self.positions_path, exist_ok=True)
        #self.quaternions = pd.DataFrame(columns=["qw", "qx", "qy", "qz"])
            self.positions_fp = open(f"{seg_plot.positions_path}/quaternion_run_{seg_plot.time_run}.csv", "w")
            self.positions_fp.write("qw,qx,qy,qz\n")
        
            

        # initialize model
        model_path = os.path.join(self.base_dir, self.model_file)
        self.model = MultiHeadUNet(heads=3, feat_dim=64, out_ch=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.ultrasound_border_mask: Image = Image.open('images/ultrasound_mask.png')

        # initialize plotting variables
        self.plot_initialized = False
        self.all_hulls_3d = []

        # logger setup
        if self.log_level.upper() not in [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]:
            raise ValueError(f"Invalid log level: {self.log_level}")
        self.logger = getLogger(__name__)
        self.logger.setLevel(self.log_level.upper())
        self.logger.info(
            f"SegmentationPlot initialized with device={self.device}, save_results={self.save_results}, log_level={self.log_level}"
        )

        # load border image from file and convert to RGBA
        # self.ultrasound_border_mask = Image.open(f"{self.images_path}/ultrasound_mask.png").convert("RGBA")



def parse_args():
    parser = argparse.ArgumentParser(description="PyClariusCast")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (cpu or cuda)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()
    return args.device, args.log_level


## main function
def main():
    device, log_level = parse_args()
    global seg_plot
    seg_plot = SegmentationPlot(
        device=device, log_level=log_level
    )
    plot_signaller = PlotSignaller()
    plot_signaller.plot_update.connect(plot_update_handler)
    seg_plot.plot_signaller = plot_signaller  # make it accessible
    load_dotenv()
    cast = pyclariuscast.Caster(
        newProcessedImage,
        newRawImage,
        newSpectrumImage,
        newImuData,
        freezeFn,
        buttonsFn,
    )
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWidget(cast)
    widget.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
