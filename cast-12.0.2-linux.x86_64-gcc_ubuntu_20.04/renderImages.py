import sys
import csv
import os
import numpy as np
from PySide6 import QtWidgets
from matplotlib.image import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def plot_images_from_quaternions(quaternions, image_path, distance=10, image_size=5, delay=1):
    """
    Plot images normal to the probe direction based on quaternion data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Load the image
    image = imread(image_path)

    for q in quaternions:
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(q)
        
        # Extract the forward direction (z-axis) of the probe
        normal_vector = rotation_matrix[:, 2]
        
        # Compute a point on the image plane (distance units away along the normal vector)
        image_center = distance * normal_vector
        
        # Define the image plane
        u = np.cross(normal_vector, [1, 0, 0])
        if np.linalg.norm(u) < 1e-6:  # Handle edge case where normal is parallel to x-axis
            u = np.cross(normal_vector, [0, 1, 0])
        u = u / np.linalg.norm(u)  # Normalize
        v = np.cross(normal_vector, u)  # Orthogonal vector
        v = v / np.linalg.norm(v)  # Normalize
        
        # Scale the image plane
        u *= image_size
        v *= image_size
        
        # Define the four corners of the image
        corners = np.array([
            image_center - u - v,
            image_center + u - v,
            image_center + u + v,
            image_center - u + v
        ])
        
        # Plot the image as a texture on the plane
        ax.plot_surface(
            np.array([corners[:, 0]]),  # X-coordinates
            np.array([corners[:, 1]]),  # Y-coordinates
            np.array([corners[:, 2]]),  # Z-coordinates
            rstride=1, cstride=1, facecolors=image, shade=False
        )

        # Pause to show the current image before plotting the next one
        plt.draw()
        plt.pause(delay)

    # Set plot limits and labels
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_images_from_quaternions_and_folder(quaternions, image_folder, distance=10, image_size=5, delay=1):
    """
    Plot images normal to the probe direction based on quaternion data and images in a folder.
    
    Args:
        quaternions: List or array of quaternions (w, x, y, z).
        image_folder: Path to the folder containing images.
        distance: Distance from the origin to place the image.
        image_size: Size of the image to plot.
        delay: Delay in seconds between plotting each image.
    """
    # Get all image paths from the folder
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) != len(quaternions):
        raise ValueError("The number of images in the folder must match the number of quaternions.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for q, image_path in zip(quaternions, image_files):
        # Load the image
        image = imread(image_path)

        # Normalize the image to [0, 1] if needed
        if image.dtype != np.float32:
            image = image / 255.0

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(q)
        
        # Extract the forward direction (z-axis) of the probe
        normal_vector = rotation_matrix[:, 2]
        
        # Compute a point on the image plane (distance units away along the normal vector)
        image_center = distance * normal_vector
        
        # Define the image plane
        u = np.cross(normal_vector, [1, 0, 0])
        if np.linalg.norm(u) < 1e-6:  # Handle edge case where normal is parallel to x-axis
            u = np.cross(normal_vector, [0, 1, 0])
        u = u / np.linalg.norm(u)  # Normalize
        v = np.cross(normal_vector, u)  # Orthogonal vector
        v = v / np.linalg.norm(v)  # Normalize
        
        # Scale the image plane
        u *= image_size
        v *= image_size
        
        # Define the four corners of the image
        corners = np.array([
            image_center - u - v,
            image_center + u - v,
            image_center + u + v,
            image_center - u + v
        ])
        
        # Create a grid for the image plane
        grid_x, grid_y = np.meshgrid(
            np.linspace(corners[0, 0], corners[2, 0], image.shape[1]),
            np.linspace(corners[0, 1], corners[2, 1], image.shape[0])
        )
        grid_z = np.linspace(corners[0, 2], corners[2, 2], image.shape[0])[:, None] + \
                 np.zeros_like(grid_x)

        # Map the image onto the plane
        ax.plot_surface(
            grid_x,  # X-coordinates
            grid_y,  # Y-coordinates
            grid_z,  # Z-coordinates
            rstride=1, cstride=1, facecolors=image, shade=False
        )

        # Pause to show the current image before plotting the next one
        plt.draw()
        plt.pause(delay)

    # Set plot limits and labels
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

class ImagePlotterApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Plotter")
        self.resize(400, 200)

        # Initialize file paths
        self.image_folder_path = "./images/2025-04-18"
        self.quaternion_file_path = "./positions/quaternion_run_2025-04-18.csv"

        # Layout
        layout = QtWidgets.QVBoxLayout()

        # Load Quaternion Button
        self.load_quaternion_button = QtWidgets.QPushButton("Load Quaternion File")
        self.load_quaternion_button.clicked.connect(self.load_quaternion_file)
        layout.addWidget(self.load_quaternion_button)

        # Load Image Folder Button
        self.load_image_folder_button = QtWidgets.QPushButton("Load Image Folder")
        self.load_image_folder_button.clicked.connect(self.load_image_folder)
        layout.addWidget(self.load_image_folder_button)

        # Run Button
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.run_plot)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def load_quaternion_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Quaternion File", "", "CSV Files (*.csv *.txt)")
        if not file_path:
            return
        self.quaternion_file_path = file_path
        print(f"Loaded Quaternion File: {file_path}")

    def load_image_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Folder", "")
        if not folder_path:
            return
        self.image_folder_path = folder_path
        print(f"Loaded Image Folder: {folder_path}")

    def run_plot(self):
        if not self.quaternion_file_path or not self.image_folder_path:
            QtWidgets.QMessageBox.warning(self, "Missing Files", "Please load a quaternion file and either an image file or an image folder.")
            return

        # Load quaternions from the file
        quaternions = self.load_quaternions_from_csv(self.quaternion_file_path)
        if quaternions is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load quaternion data.")
            return

        # Run the plot function
        if self.image_folder_path:
            try:
                plot_images_from_quaternions_and_folder(quaternions, self.image_folder_path)
            except ValueError as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
        else:
            plot_images_from_quaternions(quaternions, self.image_file_path)

    def load_quaternions_from_csv(self, path):
        quats = []
        try:
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
        except Exception as e:
            print(f"Error loading quaternion file: {e}")
            return None

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImagePlotterApp()
    window.show()
    sys.exit(app.exec())