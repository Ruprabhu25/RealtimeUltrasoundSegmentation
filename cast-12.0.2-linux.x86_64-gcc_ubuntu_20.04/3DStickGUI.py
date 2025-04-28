#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import csv

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QUrl
from PySide6.QtGui import QQuaternion, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

class ScannerWindow(Qt3DExtras.Qt3DWindow):
    def __init__(self):
        super(ScannerWindow, self).__init__()
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0

        # Camera setup
        self.camera().lens().setPerspectiveProjection(50, 16 / 9, 0.1, 1000)
        self.camera().setPosition(QVector3D(0, 0, 30))
        self.camera().setViewCenter(QVector3D(0, 0, 0))

        # Scene
        self.createScene()
        self.setRootEntity(self.rootEntity)

    def createScene(self):
        self.rootEntity = Qt3DCore.QEntity()
        self.scannerEntity = Qt3DCore.QEntity(self.rootEntity)
        
        # Load scanner.obj
        self.scanner = Qt3DRender.QSceneLoader(self.scannerEntity)
        self.scanner.setSource(QUrl.fromLocalFile("scanner.obj"))
        self.scannerEntity.addComponent(self.scanner)

        self.scannerTransform = Qt3DCore.QTransform()
        self.scannerEntity.addComponent(self.scannerTransform)
        self.updateAngle(1, 0, 0, 0)

    def updateAngle(self, qw, qx, qy, qz):
        self.qw, self.qx, self.qy, self.qz = qw, qx, qy, qz
        orientation = QQuaternion(qw, qx, qy, qz)
        
        # Apply corrections
        axisCorrection = QQuaternion.fromEulerAngles(0, 180, 90)
        modelCorrection = QQuaternion.fromEulerAngles(-90, 0, 90)
        corrected = modelCorrection * (orientation * axisCorrection)

        self.scannerTransform.setScale3D(QVector3D(100, 100, 100))
        self.scannerTransform.setRotation(corrected)

class MainWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWidget, self).__init__()
        self.setWindowTitle("Scanner Viewer with CSV Animation")

        self.scanner = ScannerWindow()
        scannerWidget = QtWidgets.QWidget.createWindowContainer(self.scanner)

        self.loadButton = QtWidgets.QPushButton("Load Quaternion File")
        self.loadButton.clicked.connect(self.load_file)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(scannerWidget)
        layout.addWidget(self.loadButton)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.quaternions = []
        self.index = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateRotation)
        self.timer.singleShot(20, self.updateCaption)

    def load_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Quaternion File", "", "CSV Files (*.csv *.txt)")
        if not file_path:
            return

        self.quaternions = self.load_quaternions_from_csv(file_path)
        if len(self.quaternions) > 0:
            self.index = 0
            self.timer.start(1000 / 35)  # 20 FPS animation

    def load_quaternions_from_csv(self, path):
        quats = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            for row in reader:
                try:
                    q = list(map(float, row[1:5]))
                    quats.append(q)
                except ValueError:
                    continue
        return np.array(quats)

    def updateRotation(self):
        if self.index >= len(self.quaternions):
            self.timer.stop()
            return

        q = self.quaternions[self.index]
        self.scanner.updateAngle(q[0], q[1], q[2], q[3])
        self.index += 1

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWidget()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
