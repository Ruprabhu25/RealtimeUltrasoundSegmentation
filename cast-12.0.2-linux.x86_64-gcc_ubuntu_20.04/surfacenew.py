"""
3DStickSurface.py
=================

Visualises a hand-held ultrasound probe (loaded from *scanner.obj*) together
with a translucent, 120° fan-shaped scan surface that is rigidly attached
to the probe and inherits its orientation automatically.
"""

import math
import sys
import csv
import numpy as np

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


# ─── tune these to your scanner.obj units ────────────────────────────────
SCANNER_SCALE    = 100.0   # uniform scale applied to everything
MODEL_NEAR_W     = 0.02    # width of fan at the probe face (model‐units)
MODEL_DEPTH      = 0.10    # how far it goes (model‐units)
MODEL_TIP_OFFSET = 0.07    # where the mesh’s tip lives (model‐units)
# ─────────────────────────────────────────────────────────────────────────


class ScannerWindow(QMainWindow):
    def __init__(self, quat_file: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Probe + Fan Visualisation")
        self.quats = self._load_quats(quat_file)
        self.idx   = 0

        # Qt3D setup
        self.view = Qt3DWindow()
        container = QtWidgets.QWidget.createWindowContainer(self.view)
        self.setCentralWidget(container)
        self.rootEntity = self._create_scene()
        self.view.setRootEntity(self.rootEntity)

        # drive at ~60 Hz
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000//60)

    def _load_quats(self, path):
        arr = []
        with open(path, newline="") as f:
            rdr = csv.reader(f)
            next(rdr, None)
            for row in rdr:
                try:
                    arr.append([float(v) for v in row[1:5]])
                except:
                    pass
        if not arr:
            raise RuntimeError("No quaternions!")
        return np.array(arr, dtype=np.float32)

    def _create_scene(self):
        root = QEntity()

        # ── camera + light ──────────────────────────────────────────────
        cam = self.view.camera()
        cam.lens().setPerspectiveProjection(35, 16/9, 0.1, 1000)
        cam.setPosition(QVector3D(0, 18, 45))
        cam.setViewCenter(QVector3D(0, 0, 0))

        lightE = QEntity(root)
        light  = QPointLight(lightE)
        light.setColor("white"); light.setIntensity(1.0)
        tL = QTransform()
        tL.setTranslation(QVector3D(0, 18, 45))
        lightE.addComponent(light)
        lightE.addComponent(tL)

        # ── scannerEntity: holds uniform scale + live rotation ───────────
        self.scannerEntity    = QEntity(root)
        self.scannerTransform = QTransform()
        self.scannerTransform.setScale(SCANNER_SCALE)
        self.scannerEntity.addComponent(self.scannerTransform)

        # ── meshEntity: the .obj probe (inherits scale+rot) ─────────────
        mE = QEntity(self.scannerEntity)
        mesh = QMesh(mE)
        mesh.setSource(QUrl.fromLocalFile("scanner.obj"))
        mat  = QPhongMaterial(mE)
        mat.setDiffuse(QColor("#C8C8C8"))
        mE.addComponent(mesh)
        mE.addComponent(mat)

        # ── fanEntity: child of scannerEntity (inherits scale+rot) ──────
        fE = QEntity(self.scannerEntity)
        # 1) move it out to the tip in **model** units
        # tf = QTransform()
        # tf.setTranslation(QVector3D(0.0, MODEL_TIP_OFFSET, 0.0))
        # fE.addComponent(tf)
        self.fanTransform = QTransform(fE)  
        self.fanTransform.setTranslation(QVector3D(0.0, MODEL_TIP_OFFSET, 0.0))
        fE.addComponent(self.fanTransform)

        # 2) build the trapezoid in model coords (Y–Z plane, X=0)
        geom   = QGeometry(fE)
        posB   = QBuffer(geom)
        idxB   = QBuffer(geom)

        nearW  = MODEL_NEAR_W
        depth  = MODEL_DEPTH
        half   = math.tan(math.radians(60))
        farW   = 2 * depth * half

        verts = np.array([
            [-nearW/2, 0.0,        0.0],  # near-left  (X, Y, Z)
            [ nearW/2, 0.0,        0.0],  # near-right
            [ farW/2,  depth,      0.0],  # far-right
            [-farW/2,  depth,      0.0],  # far-left
        ], dtype=np.float32)
        posB.setData(verts.tobytes())

        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
        idxB.setData(indices.tobytes())

        # position attr
        pAttr = QAttribute(geom)
        pAttr.setName(QAttribute.defaultPositionAttributeName())
        pAttr.setAttributeType(QAttribute.VertexAttribute)
        pAttr.setBuffer(posB)
        pAttr.setVertexBaseType(QAttribute.Float)
        pAttr.setVertexSize(3)
        pAttr.setCount(4)
        geom.addAttribute(pAttr)

        # index attr
        iAttr = QAttribute(geom)
        iAttr.setAttributeType(QAttribute.IndexAttribute)
        iAttr.setBuffer(idxB)
        iAttr.setVertexBaseType(QAttribute.UnsignedShort)
        iAttr.setVertexSize(1)
        iAttr.setCount(6)
        geom.addAttribute(iAttr)

        # renderer + translucent mat
        rend = QGeometryRenderer(fE)
        rend.setGeometry(geom)
        rend.setPrimitiveType(QGeometryRenderer.Triangles)
        fE.addComponent(rend)

        fmat = QPhongAlphaMaterial(fE)
        fmat.setDiffuse(QColor(0, 255, 255, 80))
        fE.addComponent(fmat)

        return root

    def _tick(self):
        w, x, y, z = self.quats[self.idx]
        self.idx = (self.idx + 1) % len(self.quats)
        q = QQuaternion(w, x, y, z)
        self.scannerTransform.setRotation(q)

    def resizeEvent(self, ev):  # keep aspect
        self.view.resize(self.size())
        super().resizeEvent(ev)


def main():
    app = QApplication(sys.argv)
    win = ScannerWindow("positions/quaternion_run_30.csv")
    win.resize(1280, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()