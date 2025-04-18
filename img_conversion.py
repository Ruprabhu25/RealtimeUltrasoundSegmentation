from PySide6.QtGui import QImage
import numpy as np
import matplotlib.pyplot as plt

def qimage_to_numpy_grayscale(image: QImage) -> np.ndarray:
    """Convert QImage to a NumPy grayscale array (1 channel)."""
    image = image.convertToFormat(QImage.Format_Grayscale8)
    width = image.width()
    height = image.height()

    ptr = image.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=height * image.bytesPerLine())
    arr = arr.reshape((height, image.bytesPerLine()))
    return arr[:, :width]  # Remove padding

def load_grayscale_image(image_path: str) -> np.ndarray:
    """Load image and convert to grayscale NumPy array."""
    image = QImage(image_path)
    if image.isNull():
        raise ValueError(f"Failed to load image: {image_path}")
    
    return qimage_to_numpy_grayscale(image)

def plot_grayscale_image(arr: np.ndarray, title: str = "Grayscale Image"):
    """Plot the grayscale image using matplotlib."""
    plt.imshow(arr, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "processed_image.png"  # Replace with your actual image path
    grayscale_array = load_grayscale_image(image_path)
    print("Grayscale image shape:", grayscale_array.shape)
    
    # Plotting the image
    plot_grayscale_image(grayscale_array)


