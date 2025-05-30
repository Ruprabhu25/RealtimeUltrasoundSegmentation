"""
3DStickSurface.py
=================

Visualises a hand-held ultrasound probe (loaded from *scanner.obj*) together
with a translucent, 120-degree fan-shaped scan surface that is rebuilt each
frame from the current quaternion, so it always points along the probe’s true
forward (+Z) axis.

Author  : <your name>
Requires: PySide6-Qt3D, NumPy
"""

import math
import sys
import numpy as np
import csv         # ← NEW: for robust CSV reading with header skip


# ─────────────────────────────────────────────────────────────────────────────
# Qt / PySide 6 imports – pull in the classes directly
# ─────────────────────────────────────────────────────────────────────────────
from PySide6.QtCore    import QUrl, QTimer
from PySide6.QtGui     import QColor, QQuaternion, QVector3D
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6 import QtWidgets      # ← NEW: for createWindowContainer


# ---------- Qt 3D imports (robust across PySide6/conda builds) ----------
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
# ScannerWindow – Qt3D view + live quaternion playback
# ─────────────────────────────────────────────────────────────────────────────
class ScannerWindow(QMainWindow):
    def __init__(self, quat_file: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Realtime Probe Visualisation")

        # quaternion stream: (w, x, y, z) rows
        # self.quats = np.loadtxt(quat_file, dtype=np.float32)
        self.quats = self.load_quaternions_from_csv(quat_file)
        self.q_idx = 0

        # main 3-D window
        self.view = Qt3DWindow()
        container = QtWidgets.QWidget.createWindowContainer(self.view)
        self.setCentralWidget(container)

        # build scene graph
        self.rootEntity = self.createScene()
        self.view.setRootEntity(self.rootEntity)

        # timer drives the animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateRotation)
        self.timer.start(1000 / 100)                      # ≃25 fps

    def load_quaternions_from_csv(self, path: str) -> np.ndarray:
        """Mimics the original load_quaternions_from_csv: skips header and
           takes columns 1-4 (w, x, y, z)."""
        quats = []
        with open(path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)            # skip header row
            for row in reader:
                try:
                    quats.append([float(v) for v in row[1:5]])
                except (ValueError, IndexError):
                    continue
        if not quats:
            raise ValueError(f"No valid quaternions in {path!r}")
        return np.asarray(quats, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # createScene – entire Qt3D scene
    # ─────────────────────────────────────────────────────────────────────
    def createScene(self) -> QEntity:
        root = QEntity()

        # ─── camera + light ───
        cam = self.view.camera()
        cam.lens().setPerspectiveProjection(35.0, 16/9, 0.1, 1000)
        cam.setPosition(QVector3D(0.0, 18.0, 45.0))
        cam.setViewCenter(QVector3D(0.0, 0.0, 0.0))

        lightEnt = QEntity(root)
        light    = QPointLight(lightEnt)
        light.setColor("white")
        light.setIntensity(1.0)
        lxf = QTransform()
        lxf.setTranslation(QVector3D(0.0, 18.0, 45.0))
        lightEnt.addComponent(light)
        lightEnt.addComponent(lxf)

        # ─── probe body (scanner.obj) ───
        self.scannerEntity = QEntity(root)

        mesh = QMesh(self.scannerEntity)
        mesh.setSource(QUrl.fromLocalFile("scanner.obj"))

        mat  = QPhongMaterial(self.scannerEntity)
        mat.setDiffuse(QColor("#C8C8C8"))

        xf   = QTransform()
        xf.setScale3D(QVector3D(100.0, 100.0, 100.0))
        self.scannerTransform = xf        # keep a handle for live updates

        self.scannerEntity.addComponent(mesh)
        self.scannerEntity.addComponent(mat)
        self.scannerEntity.addComponent(xf)

        # ─── scan fan (120°) – geometry buffers initialised once, data updated per frame ───
        self.near_w         = 2.0                       # width at probe face
        self.depth          = 10.0                      # projection depth
        self.fan_half_angle = math.radians(60.0)        # ±60 °
        self.tip_offset     = 5.0

        self.fanEntity = QEntity(root)

        geom        = QGeometry(self.fanEntity)
        self.fanPos = QBuffer(geom)
        self.fanIdx = QBuffer(geom)

        # placeholder: 4 vertices → all zeros (will be overwritten every frame)
        self.fanPos.setData(np.zeros((4, 3), np.float32).tobytes())
        self.fanIdx.setData(np.array([0, 1, 2, 0, 2, 3], np.uint16).tobytes())

        posAttr = QAttribute(geom)
        posAttr.setName(QAttribute.defaultPositionAttributeName())
        posAttr.setAttributeType(QAttribute.VertexAttribute)
        posAttr.setBuffer(self.fanPos)
        posAttr.setVertexBaseType(QAttribute.Float)
        posAttr.setVertexSize(3)
        posAttr.setCount(4)
        geom.addAttribute(posAttr)

        idxAttr = QAttribute(geom)
        idxAttr.setAttributeType(QAttribute.IndexAttribute)
        idxAttr.setBuffer(self.fanIdx)
        idxAttr.setVertexBaseType(QAttribute.UnsignedShort)
        idxAttr.setVertexSize(1)
        idxAttr.setCount(6)
        geom.addAttribute(idxAttr)

        fanMesh = QGeometryRenderer(self.fanEntity)
        fanMesh.setGeometry(geom)
        fanMesh.setPrimitiveType(QGeometryRenderer.Triangles)
        self.fanEntity.addComponent(fanMesh)

        fanMat = QPhongAlphaMaterial(self.fanEntity)
        fanMat.setDiffuse(QColor(0, 255, 255, 80))      # translucent cyan
        self.fanEntity.addComponent(fanMat)

        return root

    # ─────────────────────────────────────────────────────────────────────
    #  Timer callback – advance quaternion stream
    # ─────────────────────────────────────────────────────────────────────
    def updateRotation(self) -> None:
        q = self.quats[self.q_idx]
        self.q_idx = (self.q_idx + 1) % len(self.quats)
        self.setProbeOrientation(q[0], q[1], q[2], q[3])

    # ------------------------------------------------------------------ #
    def setProbeOrientation(self, w: float, x: float, y: float, z: float) -> None:
        quat = QQuaternion(w, x, y, z)
        self.scannerTransform.setRotation(quat)
        self.rebuildFan(quat)

    # ─────────────────────────────────────────────────────────────────────
    # rebuildFan – recompute fan vertices from current quaternion
    # ─────────────────────────────────────────────────────────────────────
    def rebuildFan(self, q: QQuaternion) -> None:
        # quaternion → rotation matrix (row-major)
        w, x, y, z = q.scalar(), q.x(), q.y(), q.z()
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)

        n = R[:, 1]                           # probe forward (+Z)
        n /= np.linalg.norm(n)

        # orthonormal basis (u, v) on image plane
        u = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        # centres of near & far edges
        # near_c = np.zeros(3, dtype=np.float32)
        # far_c  = n * self.depth
        near_c = n * self.tip_offset
        far_c  = n * (self.tip_offset + self.depth)

        near_w = self.near_w
        far_w  = 2 * self.depth * math.tan(self.fan_half_angle)

        verts = np.array([
            near_c - u * near_w/2,    # near-left
            near_c + u * near_w/2,    # near-right
            far_c  + u * far_w/2,     # far-right
            far_c  - u * far_w/2      # far-left
        ], dtype=np.float32)

        # upload new vertices → previous frame vanishes automatically
        self.fanPos.setData(verts.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    app = QApplication(sys.argv)
    win = ScannerWindow("./positions/quaternion_run_30.csv")
    win.resize(1280, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
