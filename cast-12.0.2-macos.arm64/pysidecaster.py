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
from model.us_unet2 import MultiHeadUNet
import cv2
import matplotlib.pyplot as plt
import pyclariuscast
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Slot
import pandas as pd
import os
from dotenv import load_dotenv
from convex_hull import (
    extract_3d_hull_from_image,
    get_rotation_center,
    update_plot_with_new_frame,
    quaternion_distance,
    MIN_ROTATION_RAD,
    MAX_ROTATION_RAD
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
        self.setWindowTitle("Clarius Cast Demo")

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
        segmentBtn = QtWidgets.QPushButton("Segment Image")
        plotBtn = QtWidgets.QPushButton("Plot")

        # try to connect/disconnect to/from the probe
        def tryConnect():
            try:
                seg_plot.frame_num = 0
            except Exception as e:
                seg_plot.logger.error(e)
            if not cast.isConnected():
                if cast.connect(ip.text(), int(port.text()), "research"):
                    self.statusBar().showMessage("Connected")
                    conn.setText("Disconnect")
                else:
                    self.statusBar().showMessage(
                        "Failed to connect to {0}".format(ip.text())
                    )
            else:
                if cast.disconnect():
                    seg_plot.logger.info("disconnected")
                    if seg_plot.save_results:
                        seg_plot.logger.info(f"size of quaternion data: seg_plot.quaternions.count()")
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
            seg_plot.segment_image = True
            seg_plot.plot = False
            self.statusBar().showMessage("Segmenting image...")

        def tryPlot():
            if seg_plot.segment_image is True:
                seg_plot.plot = True
                self.statusBar().showMessage("Plotting...")
            else:
                self.statusBar().showMessage("Press the Segment Image button first")

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
        if sys.platform.startswith("linux"):
            # unload the shared library before destroying the cast object
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
        self.cast.destroy()
        seg_plot.logger.info("trying to shutdown")
        if seg_plot.save_results:
            seg_plot.logger.info("saving quaternion data")
            seg_plot.quaternions.to_csv(
                f"{seg_plot.positions_path}/quaternion_run_{seg_plot.time_run}.csv",
                columns=["qw", "qx", "qy", "qz"],
                index=False,
            )
        seg_plot.logger.info("Shutting down plot")
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
    if bpp == 4:
        img_qt = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
        img_pil = Image.frombytes("RGBA", (width, height), image)
    else:
        img_qt = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
        img_pil = Image.frombytes("L", (width, height), image)
    try:
        if seg_plot.segment_image:
            pred, pred_img = segment_image(img_pil, image_size)
        try:
            if seg_plot.save_results:
                seg_plot.quaternions = pd.concat(
                    [
                        seg_plot.quaternions,
                        pd.DataFrame(
                            [
                                {
                                    "qw": imu[0].qw,
                                    "qx": imu[0].qx,
                                    "qy": imu[0].qy,
                                    "qz": imu[0].qz,
                                }
                            ]
                        ),
                    ]
                )
                seg_plot.logger.debug(f"saving {seg_plot.frame_num}")
                if seg_plot.segment_image:
                    pred_img.save(f"{seg_plot.images_path}/{seg_plot.time_run}/{seg_plot.frame_num}_segmented.png")
                img_pil.save(f"{seg_plot.images_path}/{seg_plot.time_run}/{seg_plot.frame_num}.png")
                seg_plot.logger.debug(f"saved {seg_plot.frame_num}")
                seg_plot.frame_num += 1

            if seg_plot.plot:
                plot_frame(imu, pred_img)
        except Exception as e:
            seg_plot.logger.error(e)

        if seg_plot.segment_image:
            img_qt_resized = display_segmented_image(img_pil, image_size, pred)
        else:
            img_qt_resized = img_qt
    except Exception as e:
        seg_plot.logger.error(e)
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
    img_resized = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0)
    img_resized = img_resized.permute(0, 3, 1, 2)
    with torch.no_grad():
        _, pred = seg_plot.model(img_resized)
        pred = pred.squeeze(0).numpy()

    return pred, Image.fromarray((pred.squeeze() * 255).astype(np.uint8))

def display_segmented_image(img_pil, image_size, pred: np.ndarray):
    # Resize original grayscale image to match segmentation output size
    img_original_resized = img_pil.convert("RGB").resize(image_size)

    # Create a transparent blue mask from the prediction
    # img_colored = apply_colormap(pred, cmap_name="plasma")
    # img_colored.show()
    # time.sleep(1000)
    try:
        print(pred)
        print(pred.shape)
    except:
        print("cant print dims")
    pred = pred.squeeze(0)
    #pred_thresholded = np.where(pred < 0.9, 0, 1)
    seg_mask = (pred * 255).astype(np.uint8)
    blue_overlay = np.zeros((128, 128, 4), dtype=np.uint8)
    blue_overlay[..., 2] = 255  # full blue
    blue_overlay[..., 3] = (seg_mask * 0.5).astype(np.uint8)  # alpha: 0–128

    # Convert both to PIL
    original_img = img_original_resized.convert("RGBA")
    overlay_img = Image.fromarray(blue_overlay, mode="RGBA")

    # Mask overlay: only blue where ultrasound border mask is white
    border_mask_np = np.array(seg_plot.ultrasound_border_mask)
    overlay_np = np.array(overlay_img)
    white_pixels = np.all(border_mask_np[..., :3] == 255, axis=-1)
    # Set blue only where mask is white, else transparent
    overlay_np[~white_pixels] = [0, 0, 0, 0]
    overlay_img = Image.fromarray(overlay_np, mode="RGBA")

    # Composite the overlay on top of the original image
    composited = Image.alpha_composite(original_img, overlay_img)

    # Combine (side by side) the composited image and the original image
    final_display = Image.new("RGBA", (128 * 2, 128))
    final_display.paste(composited, (0, 0))
    final_display.paste(original_img, (128, 0))

    # Convert to QImage for display
    return QtGui.QImage(
        final_display.tobytes("raw", "RGBA"),
        final_display.size[0],
        final_display.size[1],
        final_display.size[0] * 4,
        QtGui.QImage.Format_RGBA8888,
    )

def plot_update_handler(hull_3d, all_hulls_3d):
    if not seg_plot.plot_initialized:
        plt.ion()
        seg_plot.fig = plt.figure(figsize=(10, 10))
        seg_plot.ax = seg_plot.fig.add_subplot(111, projection="3d")
        seg_plot.ax.set_xlim(-0.5, 0.5)
        seg_plot.ax.set_ylim(-0.5, 0.5)
        seg_plot.ax.set_zlim(-0.5, 0.5)
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
    plot: bool = True # whether to plot image or not
    segment_image: bool = True  # whether to segment the image
    base_dir: str = os.getcwd()  # base directory for saving results
    model_file: str = "best_mhu-2.pth"  # model file name
    plot_signaller: PlotSignaller = None  # signaller for plot updates
    last_quat: list = None
    frame_num: int = 0


    def __post_init__(self):
        # if saving results, create directories for images and positions
        self.images_path = os.path.join(self.base_dir, "images")
        if self.save_results:
            self.frame_num = 0
            self.time_run = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            os.makedirs(os.path.join(self.images_path, self.time_run), exist_ok=True)
            self.positions_path = os.path.join(self.base_dir, "positions")
            os.makedirs(self.positions_path, exist_ok=True)
        self.quaternions = pd.DataFrame(columns=["qw", "qx", "qy", "qz"])

        # initialize model
        model_path = os.path.join(self.base_dir, self.model_file)
        self.model = MultiHeadUNet(heads=3, feat_dim=64, out_ch=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

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
        self.ultrasound_border_mask = Image.open(f"{self.images_path}/ultrasound_mask.png").convert("RGBA")


def parse_args():
    parser = argparse.ArgumentParser(description="PyClariusCast")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (cpu or cuda)",
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Whether to save results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="best_mhu-2.pth",
        help="Path to the model file",
    )

    args = parser.parse_args()
    return args.device, args.save_results, args.log_level


## main function
def main():
    device, save_results, log_level = parse_args()
    global seg_plot
    seg_plot = SegmentationPlot(
        device=device, save_results=save_results, log_level=log_level
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
    widget.resize(640, 480)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
