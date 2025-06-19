import os
import re
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from PIL import Image

# Load quaternion data: columns are qw, qx, qy, qz
quaternions = pd.read_csv(
    'cast-12.0.2-macos.arm64/positions/quaternion_run_2025-06-12_16-11-14.csv',
    skiprows=1
).values  # shape (N, 5) assuming first column is ignored
quaternions = quaternions[:, 1:]  # Exclude first column (nan or index)
print(f"Loaded {quaternions.shape[0]} quaternions.")
print(quaternions[:5])  # Preview first 5 quaternions

# Reorder columns from (qw, qx, qy, qz) to (qx, qy, qz, qw) for scipy
quaternions = quaternions[:, [1, 2, 3, 0]]

def plot_textured_plane(ax, image_path, rotation: R, center=np.zeros(3), size=1.0):
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("L")
    img = np.array(img) / 255.0  # Normalize to [0,1]

    h, w = img.shape  # Expect 128 x 128 or similar
    x = np.linspace(-size / 2, size / 2, w)
    y = np.linspace(-size / 2, size / 2, h)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)

    # Flatten grid points, apply rotation and translation
    points = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
    rotated = rotation.apply(points) + center
    x_rot = rotated[:, 0].reshape(h, w)
    y_rot = rotated[:, 1].reshape(h, w)
    z_rot = rotated[:, 2].reshape(h, w)

    # Map grayscale image to RGBA colors using matplotlib colormap
    face_colors = plt.cm.gray(img)

    ax.plot_surface(
        x_rot, y_rot, z_rot,
        rstride=1, cstride=1,
        facecolors=face_colors,
        linewidth=0, antialiased=False, shade=False
    )

# Setup matplotlib 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Load and sort images by numeric filename
base_dir = "./cast-12.0.2-macos.arm64/images/2025-06-12_16-11-14"
image_paths = os.listdir(base_dir)
image_paths = [os.path.join(base_dir, img) for img in image_paths]

def extract_number(path):
    filename = os.path.basename(path)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

image_paths = sorted(image_paths, key=extract_number)
print("First 5 sorted image paths:", image_paths[:5])

# Match number of images and quaternions
num_images = len(image_paths)
num_quaternions = quaternions.shape[0]

if num_quaternions < num_images:
    quaternions = quaternions[:num_images]
elif num_images > num_quaternions:
    image_paths = image_paths[:num_quaternions]

print(f"Using {len(image_paths)} images with {quaternions.shape[0]} quaternions.")

# Create Rotation objects from quaternions
r = R.from_quat(quaternions)

# Compute centers by rotating reference point [1,0,0]
centers = r.apply(np.array([[1, 0, 0]] * len(quaternions)))

# Plot each textured plane oriented by quaternion and placed at center
for i, (rot, img_path) in enumerate(zip(r, image_paths)):
    plot_textured_plane(ax, img_path, rot, center=centers[i], size=0.5)

ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()


