import nibabel as nib
import numpy as np
from PIL import Image
import os

# Input and output paths
nii_path = "~/Downloads/565(mask).egrapilon.nii"   # or .nii.gz
output_dir = "./cast-12.0.2-macos.arm64/images/nii_slices"
os.makedirs(output_dir, exist_ok=True)

# Load NIfTI image
img = nib.load(nii_path)
data = img.get_fdata()  # shape: (X, Y, Z) or (X, Y, Z, T)

# Normalize function
def normalize_slice(slice_2d):
    slice_2d = slice_2d - np.min(slice_2d)
    slice_2d = slice_2d / np.max(slice_2d)
    return (slice_2d * 255).astype(np.uint8)

# Loop through slices (assume axial slices along Z-axis)
for i in range(data.shape[2]):
    slice_2d = data[:, :, i]
    slice_norm = normalize_slice(slice_2d)
    img_pil = Image.fromarray(slice_norm)
    img_pil.save(os.path.join(output_dir, f"slice_{i:03d}.png"))

print(f"Saved {data.shape[2]} PNG slices to {output_dir}")
