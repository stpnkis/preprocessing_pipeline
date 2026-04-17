"""Utility functions for MANTIS preprocessing pipeline."""

import os
import cv2
import numpy as np
from typing import List, Tuple


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def list_images(directory: str) -> List[str]:
    """List image files in a directory, sorted by name."""
    if not os.path.isdir(directory):
        return []
    files = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(directory, f))
    return files


def load_image(path: str) -> np.ndarray:
    """Load an image as BGR. Raises IOError on failure."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    return img


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def save_image(path: str, image: np.ndarray):
    """Save an image, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, image)


def keypoints_to_array(keypoints: list) -> np.ndarray:
    """Serialize cv2.KeyPoint list to numpy array (N x 7)."""
    if not keypoints:
        return np.empty((0, 7), dtype=np.float64)
    return np.array([
        [kp.pt[0], kp.pt[1], kp.size, kp.angle,
         kp.response, kp.octave, kp.class_id]
        for kp in keypoints
    ], dtype=np.float64)


def array_to_keypoints(arr: np.ndarray) -> list:
    """Deserialize numpy array to cv2.KeyPoint list."""
    keypoints = []
    for row in arr:
        kp = cv2.KeyPoint(
            x=float(row[0]), y=float(row[1]), size=float(row[2]),
            angle=float(row[3]), response=float(row[4]),
            octave=int(row[5]), class_id=int(row[6])
        )
        keypoints.append(kp)
    return keypoints


def save_features(path: str, keypoints: list, descriptors: np.ndarray):
    """Save keypoints and descriptors to compressed .npz file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    kp_array = keypoints_to_array(keypoints)
    np.savez_compressed(path, keypoints=kp_array, descriptors=descriptors)


def load_features(path: str) -> Tuple[list, np.ndarray]:
    """Load keypoints and descriptors from .npz file."""
    data = np.load(path)
    keypoints = array_to_keypoints(data['keypoints'])
    descriptors = data['descriptors']
    return keypoints, descriptors


def build_mask_from_polygon(shape, polygon) -> np.ndarray:
    """Build a binary mask from polygon vertices.

    Args:
        shape:   Image shape (h, w) or (h, w, c).
        polygon: List of [x, y] vertices.

    Returns:
        Single-channel uint8 mask (255 inside polygon, 0 outside).
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask
