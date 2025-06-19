import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from dotenv import load_dotenv

load_dotenv()
# Load quaternions
quaternions = pd.read_csv(
    f"{os.environ['BASE_DIR']}/positions/quaternion_run_2025-06-12_16-11-14.csv",
    skiprows=1
).values[:, 1:]  # Skip first column
quaternions = quaternions[:, [1, 2, 3, 0]]  # Reorder to x, y, z, w
r = R.from_quat(quaternions)
centers = r.apply(np.array([[1, 0, 0]] * len(quaternions)))

# Sort image paths
image_dir = f"{os.environ['BASE_DIR']}/images/2025-06-12_16-11-14"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
def extract_number(path):
    filename = os.path.basename(path)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1
image_paths = sorted(image_paths, key=extract_number)

# Match counts
num_images = len(image_paths)
if len(r) > num_images:
    r = r[:num_images]
    centers = centers[:num_images]
else:
    image_paths = image_paths[:len(r)]

# Plot setup
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Collect 3D hulls
all_hulls_3d = []


def moving_average(points, window_size=5):
    kernel = np.ones(window_size) / window_size
    padded = np.pad(points, ((window_size//2, window_size//2), (0, 0)), mode='wrap')
    smoothed = np.array([
        np.convolve(padded[:, dim], kernel, mode='valid')
        for dim in range(points.shape[1])
    ]).T
    return smoothed


def interpolate_hull(hull, target_len):
    hull = np.array(hull)
    closed = np.vstack([hull, hull[0]])
    dists = np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    dists = np.hstack([[0], dists])
    total_len = dists[-1]
    uniform_dists = np.linspace(0, total_len, target_len + 1)[:-1]
    interp = interp1d(dists, closed, axis=0, kind='quadratic')
    interpolated = interp(uniform_dists)
    smoothed = moving_average(interpolated, window_size=5)
    return smoothed


def align_hull_to_reference(hull, ref):
    distances = np.linalg.norm(hull - ref[0], axis=1)
    min_idx = np.argmin(distances)
    return np.roll(hull, -min_idx, axis=0)


for rot, center, img_path in zip(r, centers, image_paths):
    img = Image.open(img_path).convert("L")
    img = np.array(img)

    white_pixels = np.argwhere(img > 200)
    if white_pixels.shape[0] < 3:
        continue

    h, w = img.shape
    size = 0.5
    scale_x = size / w
    scale_y = size / h

    local_2d = np.zeros((white_pixels.shape[0], 2))
    local_2d[:, 0] = (white_pixels[:, 1] - w / 2) * scale_x
    local_2d[:, 1] = ((h / 2) - white_pixels[:, 0]) * scale_y

    try:
        hull_2d = ConvexHull(local_2d)
    except:
        continue

    hull_pts_2d = local_2d[hull_2d.vertices]
    hull_pts_3d_local = np.hstack([hull_pts_2d, np.zeros((len(hull_pts_2d), 1))])
    hull_pts_3d = rot.apply(hull_pts_3d_local) + center

    all_hulls_3d.append(hull_pts_3d)

# Stitch consecutive hulls
target_points = 25
for i in range(len(all_hulls_3d) - 1):
    h1 = interpolate_hull(all_hulls_3d[i], target_points)
    h2 = interpolate_hull(all_hulls_3d[i + 1], target_points)
    h2 = align_hull_to_reference(h2, h1)

    for j in range(target_points):
        p1, p2 = h1[j], h1[(j + 1) % target_points]
        q2, q1 = h2[(j + 1) % target_points], h2[j]
        quad = np.array([p1, p2, q2, q1])
        side = Poly3DCollection([quad], facecolors='lightblue', edgecolors='k', alpha=0.8)
        ax.add_collection3d(side)

# Finalize plot
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()


