#!/usr/bin/env python

"""
real_time_surface.py
====================

Connects live IMU data from the probe (via pyimu.py callbacks) to a 3D QtScene
that renders:

  1. The scanner.obj mesh, properly rotated in real time.
  2. A translucent, 120° fan‐shaped surface “emitted” from the probe tip.
  3. Cartesian axes (X: red, Y: green, Z: blue) for reference.

Whenever new IMU quaternions arrive, the probe+fan orientation updates immediately.
"""

import ctypes
import datetime
import os
import sys
import csv
import math
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Import the Clarius “Caster” library and set up callback signatures
# (pyclariuscast.so must be in the same directory, and libcast.so on Linux)
# ─────────────────────────────────────────────────────────────────────────────
# if sys.platform.startswith("linux"):
#     # Ensure libcast.so is loaded before pyclariuscast.so
#     libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle
#     pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")
# else:
#     import pyclariuscast
if sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle  # load the libcast.so shared library
    # pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")  # load the pyclariuscast.so shared library
    ctypes.CDLL("./pyclariuscast.so", ctypes.RTLD_GLOBAL)
import pyclariuscast

# ─────────────────────────────────────────────────────────────────────────────
# PySide6 / Qt Imports
# ─────────────────────────────────────────────────────────────────────────────
from PySide6.QtCore    import QUrl, QTimer, QEvent, Slot
from PySide6.QtGui     import QColor, QQuaternion, QVector3D
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6 import QtWidgets, QtCore      # ← NEW: for createWindowContainer

# Qt3D imports (robust across PySide6/conda builds)
from PySide6 import Qt3DCore, Qt3DExtras, Qt3DRender      # ← CHANGED

# pull the classes out of the modules so the rest of the file can use them
QEntity     = Qt3DCore.Qt3DCore.QEntity
QTransform  = Qt3DCore.Qt3DCore.QTransform
QGeometry   = Qt3DCore.Qt3DCore.QGeometry
QAttribute  = Qt3DCore.Qt3DCore.QAttribute
QBuffer     = Qt3DCore.Qt3DCore.QBuffer

Qt3DWindow          = Qt3DExtras.Qt3DExtras.Qt3DWindow
QPhongMaterial      = Qt3DExtras.Qt3DExtras.QPhongMaterial
QPhongAlphaMaterial = Qt3DExtras.Qt3DExtras.QPhongAlphaMaterial

QMesh              = Qt3DRender.Qt3DRender.QMesh
QGeometryRenderer  = Qt3DRender.Qt3DRender.QGeometryRenderer
QPointLight        = Qt3DRender.Qt3DRender.QPointLight


# ─────────────────────────────────────────────────────────────────────────────
# Global: record all received quaternions into a DataFrame
# ─────────────────────────────────────────────────────────────────────────────
quaternions = pd.DataFrame(columns=['qw', 'qx', 'qy', 'qz'])


# ─────────────────────────────────────────────────────────────────────────────
# Custom Qt Events and Signaller
# ─────────────────────────────────────────────────────────────────────────────
class FreezeEvent(QEvent):
    """Emitted when the freeze state changes."""
    def __init__(self, frozen: bool):
        super().__init__(QEvent.Type(QEvent.User))
        self.frozen = frozen

class ButtonEvent(QEvent):
    """Emitted when a probe button is pressed."""
    def __init__(self, btn: int, clicks: int):
        super().__init__(QEvent.Type(QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks

class ImageEvent(QEvent):
    """Emitted when a new IMU quaternion arrives."""
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.User + 2))

class Signaller(QtCore.QObject):
    """
    Receives custom QtEvents from the native callback threads, then
    re-emits as Qt Signals to the main thread.
    """
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image  = QtCore.Signal(float, float, float, float)

    def __init__(self):
        super().__init__()
        self.qw = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0

    def event(self, evt: QEvent) -> bool:
        t = evt.type()
        if t == QEvent.Type(QEvent.User):
            self.freeze.emit(evt.frozen)
        elif t == QEvent.Type(QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif t == QEvent.Type(QEvent.User + 2):
            # New quaternion available
            self.image.emit(self.qw, self.qx, self.qy, self.qz)
        return True

signaller = Signaller()


# ─────────────────────────────────────────────────────────────────────────────
# Probe + Fan + Axes 3D Window
# ─────────────────────────────────────────────────────────────────────────────
# Tweak these until the fan “just kisses” your scanner.obj tip:
SCANNER_SCALE    = 100.0    # Uniform scale for entire probe+fan
MODEL_NEAR_W     = 0.02     # Fan width at probe face (model units)
MODEL_DEPTH      = 0.10     # Fan projection depth (model units)
MODEL_TIP_OFFSET = 0.07     # Distance from mesh origin to actual tip (model units)

class ScannerWindow(Qt3DWindow):
    """
    A Qt3DWindow that renders:
      - scanner.obj mesh (scaled + rotated via IMU quaternions)
      - a 120° trapezoidal fan attached at the probe tip
      - XYZ axes (red, green, blue) for reference
    """

    def __init__(self):
        super().__init__()
        # Camera setup
        cam = self.camera()
        cam.lens().setPerspectiveProjection(50.0, 16/9, 0.1, 1000.0)
        cam.setPosition(QVector3D(0.0, 0.0, 30.0))
        cam.setViewCenter(QVector3D(0.0, 0.0, 0.0))

        # Build the 3D scene graph
        self._build_scene()
        self.setRootEntity(self.rootEntity)

    def _build_scene(self):
        """
        Constructs:
          rootEntity
            ├─ axesEntity (three colored lines)
            └─ scannerEntity
                 ├─ scannerTransform (scale+rotation)
                 └─ meshEntity (scanner.obj)
                 └─ fanEntity (translated + trapezoid geometry)
        """
        root = QEntity()

        # ─── Light ─────────────────────────────────────────────────────
        lightEnt = QEntity(root)
        light = QPointLight(lightEnt)
        light.setColor(QColor("white"))
        light.setIntensity(1.0)
        lightTf = QTransform()
        lightTf.setTranslation(QVector3D(0.0, 18.0, 45.0))
        lightEnt.addComponent(light)
        lightEnt.addComponent(lightTf)

        # ─── Axes ──────────────────────────────────────────────────────
        self._create_axes(parent=root)

        # ─── Scanner root (scale + rotation) ─────────────────────────
        self.scannerEntity    = QEntity(root)
        self.scannerTransform = QTransform()
        self.scannerTransform.setScale(SCANNER_SCALE)
        self.scannerEntity.addComponent(self.scannerTransform)

        # ─── Scanner mesh (inherits scale+rotation) ──────────────────
        meshEntity = QEntity(self.scannerEntity)
        mesh = QMesh(meshEntity)
        mesh.setSource(QUrl.fromLocalFile("scanner.obj"))
        meshMat = QPhongMaterial(meshEntity)
        meshMat.setDiffuse(QColor("#C8C8C8"))
        meshEntity.addComponent(mesh)
        meshEntity.addComponent(meshMat)

        # ─── Fan Entity (inherits scale+rotation of scannerEntity) ───
        fanEntity = QEntity(self.scannerEntity)

        # 1) Translate fan out to the probe tip in model coordinates
        self.fanTransform = QTransform(fanEntity)
        self.fanTransform.setTranslation(QVector3D(0.0, MODEL_TIP_OFFSET, 0.0))
        fanEntity.addComponent(self.fanTransform)

        # 2) Build trapezoidal fan geometry in X–Y plane (Z=0), emitting +Y
        geom = QGeometry(fanEntity)
        posBuf = QBuffer(geom)
        idxBuf = QBuffer(geom)

        near_w = MODEL_NEAR_W
        depth  = MODEL_DEPTH
        half   = math.tan(math.radians(60.0))
        far_w  = 2.0 * depth * half

        # Define 4 vertices in (X, Y, Z) model‐space:
        #   near edge at Y=0, far edge at Y=depth, all Z=0
        vertices = np.array([
            [-near_w/2.0, 0.0,      0.0],   # near-left
            [ near_w/2.0, 0.0,      0.0],   # near-right
            [ far_w/2.0,  depth,    0.0],   # far-right
            [-far_w/2.0,  depth,    0.0],   # far-left
        ], dtype=np.float32)
        posBuf.setData(vertices.tobytes())

        # Two triangles: (0,1,2) and (0,2,3)
        indices = np.array([0, 1, 2,  0, 2, 3], dtype=np.uint16)
        idxBuf.setData(indices.tobytes())

        # Position attribute
        posAttr = QAttribute(geom)
        posAttr.setName(QAttribute.defaultPositionAttributeName())
        posAttr.setAttributeType(QAttribute.VertexAttribute)
        posAttr.setBuffer(posBuf)
        posAttr.setVertexBaseType(QAttribute.Float)
        posAttr.setVertexSize(3)
        posAttr.setCount(4)
        geom.addAttribute(posAttr)

        # Index attribute
        idxAttr = QAttribute(geom)
        idxAttr.setAttributeType(QAttribute.IndexAttribute)
        idxAttr.setBuffer(idxBuf)
        idxAttr.setVertexBaseType(QAttribute.UnsignedShort)
        idxAttr.setVertexSize(1)
        idxAttr.setCount(6)
        geom.addAttribute(idxAttr)

        # Renderer + translucent material
        fanRenderer = QGeometryRenderer(fanEntity)
        fanRenderer.setGeometry(geom)
        fanRenderer.setPrimitiveType(QGeometryRenderer.Triangles)
        fanEntity.addComponent(fanRenderer)

        fanMat = QPhongAlphaMaterial(fanEntity)
        fanMat.setDiffuse(QColor(0, 255, 255, 80))  # translucent cyan
        fanEntity.addComponent(fanMat)

        # Keep the root reference
        self.rootEntity = root

    def _create_axes(self, parent: QEntity) -> None:
        """
        Creates three line‐geometry entities for X (red), Y (green), Z (blue).
        Each line goes from (0,0,0) to (axisLength, 0, 0) etc. in model units,
        then gets scaled by SCANNER_SCALE along with everything else.
        """
        axisLength = 0.5  # one‐half model‐unit; will be scaled by SCANNER_SCALE

        # X‐axis (red)
        axisX = QEntity(parent)
        geomX = QGeometry(axisX)
        posX = QBuffer(geomX)
        idxX = QBuffer(geomX)

        vertsX = np.array([
            [0.0, 0.0, 0.0],
            [axisLength, 0.0, 0.0],
        ], dtype=np.float32)
        posX.setData(vertsX.tobytes())
        indicesX = np.array([0, 1], dtype=np.uint16)
        idxX.setData(indicesX.tobytes())

        posAttrX = QAttribute(geomX)
        posAttrX.setName(QAttribute.defaultPositionAttributeName())
        posAttrX.setAttributeType(QAttribute.VertexAttribute)
        posAttrX.setBuffer(posX)
        posAttrX.setVertexBaseType(QAttribute.Float)
        posAttrX.setVertexSize(3)
        posAttrX.setCount(2)
        geomX.addAttribute(posAttrX)

        idxAttrX = QAttribute(geomX)
        idxAttrX.setAttributeType(QAttribute.IndexAttribute)
        idxAttrX.setBuffer(idxX)
        idxAttrX.setVertexBaseType(QAttribute.UnsignedShort)
        idxAttrX.setVertexSize(1)
        idxAttrX.setCount(2)
        geomX.addAttribute(idxAttrX)

        rendererX = QGeometryRenderer(axisX)
        rendererX.setGeometry(geomX)
        rendererX.setPrimitiveType(QGeometryRenderer.Lines)
        axisX.addComponent(rendererX)

        matX = QPhongMaterial(axisX)
        matX.setDiffuse(QColor("red"))
        axisX.addComponent(matX)

        # Y‐axis (green)
        axisY = QEntity(parent)
        geomY = QGeometry(axisY)
        posY = QBuffer(geomY)
        idxY = QBuffer(geomY)

        vertsY = np.array([
            [0.0, 0.0, 0.0],
            [0.0, axisLength, 0.0],
        ], dtype=np.float32)
        posY.setData(vertsY.tobytes())
        indicesY = np.array([0, 1], dtype=np.uint16)
        idxY.setData(indicesY.tobytes())

        posAttrY = QAttribute(geomY)
        posAttrY.setName(QAttribute.defaultPositionAttributeName())
        posAttrY.setAttributeType(QAttribute.VertexAttribute)
        posAttrY.setBuffer(posY)
        posAttrY.setVertexBaseType(QAttribute.Float)
        posAttrY.setVertexSize(3)
        posAttrY.setCount(2)
        geomY.addAttribute(posAttrY)

        idxAttrY = QAttribute(geomY)
        idxAttrY.setAttributeType(QAttribute.IndexAttribute)
        idxAttrY.setBuffer(idxY)
        idxAttrY.setVertexBaseType(QAttribute.UnsignedShort)
        idxAttrY.setVertexSize(1)
        idxAttrY.setCount(2)
        geomY.addAttribute(idxAttrY)

        rendererY = QGeometryRenderer(axisY)
        rendererY.setGeometry(geomY)
        rendererY.setPrimitiveType(QGeometryRenderer.Lines)
        axisY.addComponent(rendererY)

        matY = QPhongMaterial(axisY)
        matY.setDiffuse(QColor("green"))
        axisY.addComponent(matY)

        # Z‐axis (blue)
        axisZ = QEntity(parent)
        geomZ = QGeometry(axisZ)
        posZ = QBuffer(geomZ)
        idxZ = QBuffer(geomZ)

        vertsZ = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, axisLength],
        ], dtype=np.float32)
        posZ.setData(vertsZ.tobytes())
        indicesZ = np.array([0, 1], dtype=np.uint16)
        idxZ.setData(indicesZ.tobytes())

        posAttrZ = QAttribute(geomZ)
        posAttrZ.setName(QAttribute.defaultPositionAttributeName())
        posAttrZ.setAttributeType(QAttribute.VertexAttribute)
        posAttrZ.setBuffer(posZ)
        posAttrZ.setVertexBaseType(QAttribute.Float)
        posAttrZ.setVertexSize(3)
        posAttrZ.setCount(2)
        geomZ.addAttribute(posAttrZ)

        idxAttrZ = QAttribute(geomZ)
        idxAttrZ.setAttributeType(QAttribute.IndexAttribute)
        idxAttrZ.setBuffer(idxZ)
        idxAttrZ.setVertexBaseType(QAttribute.UnsignedShort)
        idxAttrZ.setVertexSize(1)
        idxAttrZ.setCount(2)
        geomZ.addAttribute(idxAttrZ)

        rendererZ = QGeometryRenderer(axisZ)
        rendererZ.setGeometry(geomZ)
        rendererZ.setPrimitiveType(QGeometryRenderer.Lines)
        axisZ.addComponent(rendererZ)

        matZ = QPhongMaterial(axisZ)
        matZ.setDiffuse(QColor("blue"))
        axisZ.addComponent(matZ)

    def updateAngle(self, qw: float, qx: float, qy: float, qz: float) -> None:
        """
        Called whenever a new IMU quaternion arrives (via Signaller).
        Applies the same “axisCorrection” and “modelCorrection” from pyimu.py
        to orient scannerEntity correctly.
        """
        # Base orientation from IMU:
        baseOri = QQuaternion(qw, qx, qy, qz)

        # These two corrections come from pyimu.py’s addTransform():
        axisCorrection  = QQuaternion.fromEulerAngles(0.0, 180.0, 90.0)
        modelCorrection = QQuaternion.fromEulerAngles(-90.0, 0.0, 90.0)

        # Combine to get the final rotation:
        modelRot = baseOri * axisCorrection
        finalOri = modelCorrection * modelRot

        # Apply to the scannerTransform (fan+mesh inherit it):
        self.scannerTransform.setRotation(finalOri)


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Window: manages UI + connections to Caster callbacks
# ─────────────────────────────────────────────────────────────────────────────
class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real‐Time Probe + Fan Viewer")
        self.cast = cast

        # Central widget & layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # IP/Port inputs + Connect/Quit buttons
        ipLine  = QtWidgets.QLineEdit("192.168.1.1")
        ipLine.setInputMask("000.000.000.000")
        portLine = QtWidgets.QLineEdit("5828")
        portLine.setInputMask("00000")
        connectBtn = QtWidgets.QPushButton("Connect")
        quitBtn    = QtWidgets.QPushButton("Quit")

        # Try to connect/disconnect logic
        def tryConnect():
            if not cast.isConnected():
                if cast.connect(ipLine.text(), int(portLine.text()), "research"):
                    self.statusBar().showMessage("Connected to " + ipLine.text())
                    connectBtn.setText("Disconnect")
                else:
                    self.statusBar().showMessage("Failed to connect")
            else:
                if cast.disconnect():
                    self.statusBar().showMessage("Disconnected")
                    connectBtn.setText("Connect")
                else:
                    self.statusBar().showMessage("Failed to disconnect")

        connectBtn.clicked.connect(tryConnect)
        quitBtn.clicked.connect(self.shutdown)

        # 3D ScannerWindow embedded in a QWidget container
        self.scannerWin = ScannerWindow()
        scannerContainer = QtWidgets.QWidget.createWindowContainer(self.scannerWin)

        # Layout arrangement
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(scannerContainer, stretch=1)
        layout.addWidget(ipLine)
        layout.addWidget(portLine)
        layout.addWidget(connectBtn)
        layout.addWidget(quitBtn)
        central.setLayout(layout)

        # Connect Signaller signals to slots
        signaller.freeze.connect(self.onFreeze)
        signaller.button.connect(self.onButton)
        signaller.image.connect(self.onImage)

        # Initialize Caster
        home = os.path.expanduser("~")
        if cast.init(home, 640, 480):
            self.statusBar().showMessage("Caster initialized")
        else:
            self.statusBar().showMessage("Caster initialization failed")

    @Slot(bool)
    def onFreeze(self, frozen: bool):
        if frozen:
            self.statusBar().showMessage("Image Stream: Frozen")
        else:
            self.statusBar().showMessage("Image Stream: Running")

    @Slot(int, int)
    def onButton(self, btn: int, clicks: int):
        self.statusBar().showMessage(f"Button {btn} pressed ({clicks} clicks)")

    @Slot(float, float, float, float)
    def onImage(self, qw: float, qx: float, qy: float, qz: float):
        # Update orientation of the probe + fan
        self.scannerWin.updateAngle(qw, qx, qy, qz)

    @Slot()
    def shutdown(self):
        # Properly destroy Caster and save quaternion log
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
        self.cast.destroy()
        global quaternions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        quaternions.to_csv(f"./positions/quaternion_live_{timestamp}.csv", index=False)
        QApplication.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Called by Caster whenever a new “processed” image + IMU data arrives.
# We only care about the IMU quaternion so we post an ImageEvent.
# ─────────────────────────────────────────────────────────────────────────────
def newProcessedImage(image, width, height, bpp, micronsPerPixel, timestamp, angle, imu):
    if len(imu) > 0:
        # Grab the first IMU quaternion (assuming imu[0].qw, .qx, .qy, .qz exist)
        signaller.qw = imu[0].qw
        signaller.qx = imu[0].qx
        signaller.qy = imu[0].qy
        signaller.qz = imu[0].qz

        # Fire a custom Qt Event so the Signaller can re‐emit as a signal
        evt = ImageEvent()
        QtCore.QCoreApplication.postEvent(signaller, evt)

        # Also append to our global log
        global quaternions
        new_row = pd.DataFrame([{
            'qw': imu[0].qw,
            'qx': imu[0].qx,
            'qy': imu[0].qy,
            'qz': imu[0].qz
        }])
        quaternions = pd.concat([quaternions, new_row], ignore_index=True)
    return


def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    return


def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


def newImuData(imu):
    return


def freezeFn(frozen: bool):
    evt = FreezeEvent(frozen)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


def buttonsFn(button: int, clicks: int):
    evt = ButtonEvent(button, clicks)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


def main():
    # Instantiate Caster with our callbacks
    cast = pyclariuscast.Caster(
        newProcessedImage,
        newRawImage,
        newSpectrumImage,
        newImuData,
        freezeFn,
        buttonsFn
    )

    app = QApplication(sys.argv)
    widget = MainWidget(cast)
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
