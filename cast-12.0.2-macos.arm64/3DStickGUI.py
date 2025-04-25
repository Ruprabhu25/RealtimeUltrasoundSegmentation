import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import csv

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

class QuaternionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quaternion Stick Viewer")

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.load_button = QPushButton("Load Quaternion File")
        self.load_button.clicked.connect(self.load_file)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.load_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.quaternions = []
        self.line = None
        self.ani = None

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Quaternion File", "", "CSV Files (*.csv *.txt)")
        if not file_path:
            return

        self.quaternions = self.load_quaternions_from_csv(file_path)
        if len(self.quaternions) > 0:
            self.start_animation()

    def load_quaternions_from_csv(self, path):
        quats = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip header
            for row in reader:
                try:
                    # Parse qw, qx, qy, qz from index 1-4
                    q = list(map(float, row[1:5]))
                    quats.append(q)
                except ValueError:
                    continue
        return np.array(quats)

    def start_animation(self):
        if self.ani:
            self.ani.event_source.stop()

        self.ax.clear()
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.line, = self.ax.plot([], [], [], lw=3, color='blue')

        self.ani = FuncAnimation(self.figure, self.update, frames=len(self.quaternions), interval=50, blit=False)
        self.canvas.draw()

    def update(self, i):
        q = self.quaternions[i]
        R = quaternion_to_rotation_matrix(q)

        start = np.array([0, 0, 0])
        end = R @ np.array([0, 0, 1])

        self.line.set_data([start[0], end[0]], [start[1], end[1]])
        self.line.set_3d_properties([start[2], end[2]])
        return self.line,


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = QuaternionViewer()
    viewer.show()
    sys.exit(app.exec())

