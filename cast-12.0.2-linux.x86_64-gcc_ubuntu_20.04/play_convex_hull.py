#!/usr/bin/env python3
"""
surface_with_masks.py

Play back probe motion + fan + overlayed mask contours,
then at the end display the 3D convex hull of all contours.

No CLI args—paths are hard-coded.
"""

import sys
import os
import math
import csv
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from PySide6.QtCore    import QUrl, QTimer
from PySide6.QtGui     import QColor, QQuaternion, QVector3D
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6 import QtWidgets
from PySide6 import Qt3DCore, Qt3DExtras, Qt3DRender

# ─── hard-coded data paths ───────────────────────────────────────────────
QUAT_PATH  = "./positions/quaternion_run_2025-06-12_16-11-14.csv"
IMAGES_DIR = "./images/2025-06-12_16-11-14"
# ─────────────────────────────────────────────────────────────────────────

# ─── visualization constants ────────────────────────────────────────────
SCANNER_SCALE     = 100.0    # uniform scale for probe+fan+contours
MODEL_NEAR_W      = 0.02     # fan width at probe face (model units)
MODEL_DEPTH       = 0.10     # fan projection depth (model units)
MODEL_TIP_OFFSET  = 0.07     # tip offset from probe origin (model units)
MASK_THRESHOLD    = 200      # pixel brightness > threshold = mask
FRAME_RATE_HZ     = 20       # playback speed, Hz
# ─────────────────────────────────────────────────────────────────────────

# shorten Qt3D type names
QEntity             = Qt3DCore.Qt3DCore.QEntity
QTransform          = Qt3DCore.Qt3DCore.QTransform
QGeometry           = Qt3DCore.Qt3DCore.QGeometry
QAttribute          = Qt3DCore.Qt3DCore.QAttribute
QBuffer             = Qt3DCore.Qt3DCore.QBuffer

Qt3DWindow          = Qt3DExtras.Qt3DExtras.Qt3DWindow
QPhongMaterial      = Qt3DExtras.Qt3DExtras.QPhongMaterial
QPhongAlphaMaterial = Qt3DExtras.Qt3DExtras.QPhongAlphaMaterial

QMesh               = Qt3DRender.Qt3DRender.QMesh
QGeometryRenderer   = Qt3DRender.Qt3DRender.QGeometryRenderer
QPointLight         = Qt3DRender.Qt3DRender.QPointLight


class SurfaceWithMasks(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Probe + Fan + Mask Contours")

        # Load data
        self.quats        = self._load_quats(QUAT_PATH)
        self.hulls_local  = self._load_mask_hulls(IMAGES_DIR)
        self.n_frames     = min(len(self.quats), len(self.hulls_local))
        self.quats        = self.quats[:self.n_frames]
        self.hulls_local  = self.hulls_local[:self.n_frames]
        self.idx          = 0
        self.world_points = []  # accumulate for global hull

        # Qt3D setup
        self.view = Qt3DWindow()
        container = QtWidgets.QWidget.createWindowContainer(self.view)
        self.setCentralWidget(container)
        self.rootEntity = self._create_scene()
        self.view.setRootEntity(self.rootEntity)

        # start timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000/FRAME_RATE_HZ))

    def _load_quats(self, path):
        arr = []
        with open(path, newline="") as f:
            rd = csv.reader(f)
            next(rd, None)
            for row in rd:
                try:
                    arr.append([float(row[1]),float(row[2]),float(row[3]),float(row[4])])
                except:
                    pass
        return np.array(arr, dtype=np.float32)

    def _load_mask_hulls(self, folder):
        # read mask images, compute convex-hull vertices in fan-local X–Y plane
        files = sorted(
            [os.path.join(folder,f) for f in os.listdir(folder)
             if f.lower().endswith((".png",".jpg",".tif"))],
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0] or 0)
        )
        hulls = []
        for path in files:
            img = Image.open(path).convert("L")
            arr = np.array(img)
            pts = np.argwhere(arr > MASK_THRESHOLD)
            if pts.shape[0] < 3:
                hulls.append(np.zeros((0,3),dtype=np.float32))
                continue
            h, w = arr.shape
            sx = MODEL_NEAR_W / w
            sy = MODEL_DEPTH   / h
            local2d = np.empty((pts.shape[0],2), dtype=np.float32)
            local2d[:,0] = (pts[:,1] - w/2)*sx
            local2d[:,1] = (h - pts[:,0])*sy
            try:
                hull2d = ConvexHull(local2d)
                pts2d = local2d[hull2d.vertices]
            except:
                pts2d = local2d
            # lift into X–Y plane at Z=0
            hulls.append(np.hstack([pts2d, np.zeros((len(pts2d),1),dtype=np.float32)]))
        return hulls

    def _create_scene(self):
        root = QEntity()

        # ─── camera + light ──────────────────────────────────────────────
        cam = self.view.camera()
        cam.lens().setPerspectiveProjection(35,16/9,0.1,1000)
        cam.setPosition(QVector3D(0,18,45))
        cam.setViewCenter(QVector3D(0,0,0))

        lightE = QEntity(root)
        light  = QPointLight(lightE)
        light.setColor("white"); light.setIntensity(1.0)
        tL = QTransform(); tL.setTranslation(QVector3D(0,18,45))
        lightE.addComponent(light); lightE.addComponent(tL)

        # ─── scanner root (scale + rotation) ─────────────────────────────
        self.scannerEnt   = QEntity(root)
        self.scannerTrans = QTransform()
        self.scannerTrans.setScale(SCANNER_SCALE)
        self.scannerEnt.addComponent(self.scannerTrans)

        # ─── probe mesh ───────────────────────────────────────────────────
        mE = QEntity(self.scannerEnt)
        mesh = QMesh(mE)
        mesh.setSource(QUrl.fromLocalFile("scanner.obj"))
        mat  = QPhongMaterial(mE); mat.setDiffuse(QColor("#C8C8C8"))
        mE.addComponent(mesh); mE.addComponent(mat)

        # ─── fan geometry ─────────────────────────────────────────────────
        self.fanEnt   = QEntity(self.scannerEnt)
        self.fanTrans = QTransform(self.fanEnt)
        self.fanTrans.setTranslation(QVector3D(0, MODEL_TIP_OFFSET, 0))
        self.fanEnt.addComponent(self.fanTrans)

        geom = QGeometry(self.fanEnt)
        posB = QBuffer(geom); idxB = QBuffer(geom)
        near = MODEL_NEAR_W; depth = MODEL_DEPTH
        half = math.tan(math.radians(60.0)); far = 2*depth*half
        verts = np.array([
            [-near/2, 0.0, 0.0],
            [ near/2, 0.0, 0.0],
            [ far/2, depth,0.0],
            [-far/2, depth,0.0],
        ], dtype=np.float32)
        posB.setData(verts.tobytes())
        idxB.setData(np.array([0,1, 0,2, 2,3],dtype=np.uint16).tobytes())

        pA = QAttribute(geom)
        pA.setName(QAttribute.defaultPositionAttributeName())
        pA.setAttributeType(QAttribute.VertexAttribute)
        pA.setBuffer(posB)
        pA.setVertexBaseType(QAttribute.Float)
        pA.setVertexSize(3)
        pA.setCount(4)
        geom.addAttribute(pA)

        iA = QAttribute(geom)
        iA.setAttributeType(QAttribute.IndexAttribute)
        iA.setBuffer(idxB)
        iA.setVertexBaseType(QAttribute.UnsignedShort)
        iA.setVertexSize(1)
        iA.setCount(6)
        geom.addAttribute(iA)

        fanR = QGeometryRenderer(self.fanEnt)
        fanR.setGeometry(geom)
        fanR.setPrimitiveType(QGeometryRenderer.Triangles)
        self.fanEnt.addComponent(fanR)

        fanM = QPhongAlphaMaterial(self.fanEnt)
        fanM.setDiffuse(QColor(0,255,255,80))
        self.fanEnt.addComponent(fanM)

        return root

    def _tick(self):
        if self.idx >= self.n_frames:
            self.timer.stop()
            self._show_global_hull()
            return

        # update probe + fan orientation
        qw,qx,qy,qz = self.quats[self.idx]
        quat = QQuaternion(qw, qx, qy, qz)
        self.scannerTrans.setRotation(quat)

        # get this frame's local hull
        local_pts = self.hulls_local[self.idx]
        if local_pts.shape[0] > 1:
            # compute world coords: (local + tip_offset) → rotated → scaled
            offset = np.array([0, MODEL_TIP_OFFSET, 0], dtype=np.float32)
            pts_model = local_pts + offset
            world_pts = []
            for p in pts_model:
                v = QVector3D(p[0], p[1], p[2])
                v = quat.rotatedVector(v)
                v *= SCANNER_SCALE
                world_pts.append((v.x(), v.y(), v.z()))
            world_pts = np.array(world_pts, dtype=np.float32)
            self.world_points.append(world_pts)

            # create a persistent contour entity
            ent = QEntity(self.rootEntity)
            geom = QGeometry(ent)
            bufP = QBuffer(geom); bufI = QBuffer(geom)
            bufP.setData(world_pts.tobytes())
            # closed loop lines
            N = len(world_pts)
            idxs = np.empty(2*N, dtype=np.uint16)
            for i in range(N):
                idxs[2*i]   = i
                idxs[2*i+1] = (i+1)%N
            bufI.setData(idxs.tobytes())

            pAttr = QAttribute(geom)
            pAttr.setName(QAttribute.defaultPositionAttributeName())
            pAttr.setAttributeType(QAttribute.VertexAttribute)
            pAttr.setBuffer(bufP)
            pAttr.setVertexBaseType(QAttribute.Float)
            pAttr.setVertexSize(3)
            pAttr.setCount(N)
            geom.addAttribute(pAttr)

            iAttr = QAttribute(geom)
            iAttr.setAttributeType(QAttribute.IndexAttribute)
            iAttr.setBuffer(bufI)
            iAttr.setVertexBaseType(QAttribute.UnsignedShort)
            iAttr.setVertexSize(1)
            iAttr.setCount(2*N)
            geom.addAttribute(iAttr)

            rend = QGeometryRenderer(ent)
            rend.setGeometry(geom)
            rend.setPrimitiveType(QGeometryRenderer.Lines)
            ent.addComponent(rend)

            mat = QPhongMaterial(ent)
            mat.setDiffuse(QColor("red"))
            ent.addComponent(mat)

        self.idx += 1

    def _show_global_hull(self):
        # flatten all world points
        all_pts = np.vstack(self.world_points)
        if all_pts.shape[0] < 4:
            return
        hull = ConvexHull(all_pts)
        simplices = hull.simplices  # (M,3) triangles

        ent = QEntity(self.rootEntity)
        geom = QGeometry(ent)
        bufP = QBuffer(geom); bufI = QBuffer(geom)
        bufP.setData(all_pts.astype(np.float32).tobytes())

        # flatten triangles
        idxs = simplices.flatten().astype(np.uint32)
        bufI.setData(idxs.tobytes())

        pAttr = QAttribute(geom)
        pAttr.setName(QAttribute.defaultPositionAttributeName())
        pAttr.setAttributeType(QAttribute.VertexAttribute)
        pAttr.setBuffer(bufP)
        pAttr.setVertexBaseType(QAttribute.Float)
        pAttr.setVertexSize(3)
        pAttr.setCount(len(all_pts))
        geom.addAttribute(pAttr)

        iAttr = QAttribute(geom)
        iAttr.setAttributeType(QAttribute.IndexAttribute)
        iAttr.setBuffer(bufI)
        iAttr.setVertexBaseType(QAttribute.UnsignedInt)
        iAttr.setVertexSize(1)
        iAttr.setCount(len(idxs))
        geom.addAttribute(iAttr)

        rend = QGeometryRenderer(ent)
        rend.setGeometry(geom)
        rend.setPrimitiveType(QGeometryRenderer.Triangles)
        ent.addComponent(rend)

        mat = QPhongAlphaMaterial(ent)
        mat.setDiffuse(QColor(0,255,0,100))
        ent.addComponent(mat)


def main():
    app = QApplication(sys.argv)
    win = SurfaceWithMasks()
    win.resize(1280, 720)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
