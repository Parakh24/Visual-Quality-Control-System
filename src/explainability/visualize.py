import cv2
import numpy as np
import os


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image: Original image as numpy array (H, W, 3), values 0-255 or 0-1
        heatmap: 2D heatmap (H, W), values 0-1
        alpha: Transparency factor
        colormap: OpenCV colormap

    Returns:
        overlayed image (H, W, 3)
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def save_overlay(image, heatmap, output_path, alpha=0.4):
    """
    Save overlay image to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    cv2.imwrite(output_path, overlay)
    return output_path
