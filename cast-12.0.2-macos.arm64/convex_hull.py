import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from dotenv import load_dotenv
import cv2
from scipy.spatial.transform import Rotation as R

from math import radians

# Configurable minimum rotation (in degrees)
MIN_ROTATION_DEG = 1  # ignore frames with < 2° change
MAX_ROTATION_DEG = 10.0
MIN_ROTATION_RAD = radians(MIN_ROTATION_DEG)
MAX_ROTATION_RAD = radians(MAX_ROTATION_DEG)


def get_rotation_center(quat):
    """Convert quaternion to rotation and calculate center."""
    rotation = R.from_quat(quat)
    center = rotation.apply(
        np.array([[1, 0, 0]])
    )
    print(f"Rotation: {rotation}, Center: {center}")
    return rotation, center


def load_quaternions_and_centers(quat_path):
    quats = pd.read_csv(quat_path, skiprows=1).values[:, 1:]
    quats = quats[:, [1, 2, 3, 0]]  # reorder to x, y, z, w
    rotations = R.from_quat(quats)
    centers = rotations.apply(np.array([[1, 0, 0]] * len(quats)))
    return rotations, centers


def get_sorted_image_paths(image_dir):
    def extract_number(path):
        filename = os.path.basename(path)
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else -1

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    return sorted(image_paths, key=extract_number)


def clean_and_extract_largest_contour(img, white_thresh=200, min_area=100):
    """Threshold, clean, and find the largest contour."""
    # Convert to binary mask
    _, binary = cv2.threshold(img, white_thresh, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Filter by area and pick the largest valid one
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not valid_contours:
        return None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    return largest_contour


def moving_average(points, window_size=5):
    kernel = np.ones(window_size) / window_size
    padded = np.pad(points, ((window_size // 2, window_size // 2), (0, 0)), mode="wrap")
    smoothed = np.array(
        [
            np.convolve(padded[:, dim], kernel, mode="valid")
            for dim in range(points.shape[1])
        ]
    ).T
    return smoothed


def interpolate_hull(hull, target_len):
    hull = np.array(hull)
    closed = np.vstack([hull, hull[0]])
    dists = np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    dists = np.hstack([[0], dists])
    total_len = dists[-1]
    uniform_dists = np.linspace(0, total_len, target_len + 1)[:-1]
    interp = interp1d(dists, closed, axis=0, kind="quadratic")
    interpolated = interp(uniform_dists)
    return moving_average(interpolated, window_size=5)


def align_hull_to_reference(hull, ref):
    distances = np.linalg.norm(hull - ref[0], axis=1)
    min_idx = np.argmin(distances)
    return np.roll(hull, -min_idx, axis=0)


def load_image_as_grayscale(img_path):
    """Load an image and convert it to grayscale."""
    img = Image.open(img_path).convert("L")
    return np.array(img)


def extract_3d_hull_from_image(img, rotation, center=None, size=0.5, white_thresh=200, min_area=100):
    contour = clean_and_extract_largest_contour(img, white_thresh, min_area)
    if contour is None or len(contour) < 3:
        return None

    h, w = img.shape
    scale_x = size / w
    scale_y = size / h

    # Convert contour to 2D scaled space
    contour = contour.squeeze()  # (N, 2)
    scaled_2d = np.zeros_like(contour, dtype=np.float32)
    scaled_2d[:, 0] = (contour[:, 0] - w / 2) * scale_x
    scaled_2d[:, 1] = ((h / 2) - contour[:, 1]) * scale_y  # Flip Y

    # Get convex hull
    try:
        hull_2d = ConvexHull(scaled_2d)
    except:
        return None

    hull_pts_2d = scaled_2d[hull_2d.vertices]

    # Ensure CCW
    if cv2.contourArea(hull_pts_2d.astype(np.float32)) < 0:
        hull_pts_2d = hull_pts_2d[::-1]

    # Convert to 3D (add z=0)
    hull_pts_3d_local = np.hstack([hull_pts_2d, np.zeros((len(hull_pts_2d), 1))])

    # Rotate entire hull
    rotated_hull = rotation.apply(hull_pts_3d_local)

    # Anchor the bottom of the rotated hull to the origin
    lowest_z_idx = np.argmin(rotated_hull[:, 2])  # Find point with lowest z
    anchor_point = rotated_hull[lowest_z_idx]
    anchored_hull = rotated_hull - anchor_point  # Translate so that point lies at (0, 0, 0)

    return anchored_hull


# debugging
def visualize_contours(img, contour):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    plt.imshow(vis)
    plt.title("Selected Contour")
    plt.axis("off")
    plt.show()

def stitch_and_plot_hulls(ax, all_hulls_3d, target_points=25):
    smoothed_hulls = []
    for hull in all_hulls_3d:
        smoothed_hulls.append(interpolate_hull(hull, target_points))

    for i in range(len(smoothed_hulls) - 1):
        h1 = smoothed_hulls[i]
        h2 = smoothed_hulls[i + 1]
        h2 = align_hull_to_reference(h2, h1)

        for j in range(target_points):
            p1, p2 = h1[j], h1[(j + 1) % target_points]
            q2, q1 = h2[(j + 1) % target_points], h2[j]
            quad = np.array([p1, p2, q2, q1])
            side = Poly3DCollection(
                [quad],
                facecolors="lightblue",
                edgecolors="none",   # no black edges
                alpha=0.8
            )
            ax.add_collection3d(side)

    # -------- Cap the bottom (first hull) --------
    bottom = smoothed_hulls[0]
    bottom_face = Poly3DCollection(
        [bottom],
        facecolors="lightblue",
        edgecolors="none",
        alpha=0.8
    )
    ax.add_collection3d(bottom_face)

    # -------- Cap the top (last hull) --------
    top = smoothed_hulls[-1]
    top_face = Poly3DCollection(
        [top[::-1]],  # reverse orientation so normal faces outward
        facecolors="lightblue",
        edgecolors="none",
        alpha=0.8
    )
    ax.add_collection3d(top_face)



def update_plot_with_new_frame(ax, hull_3d, all_hulls_3d, target_points=25):
    """Update the plot with a new quaternion, center, and image frame."""
    if hull_3d is None:
        return

    if all_hulls_3d:
        h1 = interpolate_hull(all_hulls_3d[-1], target_points)
        h2 = interpolate_hull(hull_3d, target_points)
        h2 = align_hull_to_reference(h2, h1)

        for j in range(target_points):
            p1, p2 = h1[j], h1[(j + 1) % target_points]
            q2, q1 = h2[(j + 1) % target_points], h2[j]
            quad = np.array([p1, p2, q2, q1])
            side = Poly3DCollection(
                [quad], facecolors="lightpink", edgecolors="none", alpha=1
            )
            ax.add_collection3d(side)

    all_hulls_3d.append(hull_3d)

def quaternion_distance(q1, q2):
    """Compute the angular difference (in radians) between two quaternions."""
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(dot_product)

def main():
    load_dotenv()
    base_dir = os.environ["BASE_DIR"]
    quat_path = f"{base_dir}/positions/quaternion_run_2025-06-12_16-11-14.csv"
    image_dir = f"{base_dir}/images/2025-06-12_16-11-14"

    # Load data
    rotations, centers = load_quaternions_and_centers(quat_path)
    image_paths = get_sorted_image_paths(image_dir)

    # Match counts
    num_images = len(image_paths)
    if len(rotations) > num_images:
        rotations = rotations[:num_images]
        centers = centers[:num_images]
    else:
        image_paths = image_paths[: len(rotations)]

    # Extract hulls
    all_hulls_3d = []
    for rot, center, img_path in zip(rotations, centers, image_paths):
        img = load_image_as_grayscale(img_path)
        hull_3d = extract_3d_hull_from_image(img, rot, center)
        if hull_3d is not None:
            all_hulls_3d.append(hull_3d)

    # Plot setup
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    stitch_and_plot_hulls(ax, all_hulls_3d, target_points=50)

    # Finalize plot
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(0, 0, 0, color='red', s=50, label='Origin')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
