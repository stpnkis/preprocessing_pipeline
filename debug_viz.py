"""Debug visualization helpers for MANTIS preprocessing pipeline."""

import os
import cv2
import numpy as np
from typing import Optional, Dict


def draw_keypoints(image_gray: np.ndarray, keypoints: list,
                   color=(0, 255, 0)) -> np.ndarray:
    """Draw rich keypoints on an image."""
    vis = (cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
           if len(image_gray.shape) == 2 else image_gray.copy())
    return cv2.drawKeypoints(
        vis, keypoints, None, color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_matches(cur_gray: np.ndarray, cur_kp: list,
                 ref_gray: np.ndarray, ref_kp: list,
                 matches: list, max_display: int = 200) -> np.ndarray:
    """Draw feature matches (current on left, reference on right).

    matches: queryIdx → cur_kp, trainIdx → ref_kp.
    """
    display = matches[:max_display]
    return cv2.drawMatches(
        cur_gray, cur_kp, ref_gray, ref_kp, display, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_orientation_comparison(reference: np.ndarray,
                                candidate_0: np.ndarray,
                                candidate_180: np.ndarray,
                                score_0: float,
                                score_180: float) -> np.ndarray:
    """Side-by-side comparison of reference, 0° candidate, 180° candidate."""
    h, w = reference.shape[:2]

    def _to_bgr(img):
        return (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if len(img.shape) == 2 else img.copy())

    # Scale for reasonable display
    max_h = min(h, 480)
    scale = max_h / h
    tw = int(w * scale)

    ref_s = cv2.resize(_to_bgr(reference), (tw, max_h))
    c0_s = cv2.resize(_to_bgr(candidate_0), (tw, max_h))
    c180_s = cv2.resize(_to_bgr(candidate_180), (tw, max_h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(ref_s, "Reference", (8, 22), font, 0.55, (0, 255, 255), 2)
    cv2.putText(c0_s, f"0deg NCC={score_0:.3f}", (8, 22),
                font, 0.55, (0, 255, 0), 2)
    cv2.putText(c180_s, f"180deg NCC={score_180:.3f}", (8, 22),
                font, 0.55, (0, 255, 0), 2)

    # Green border on the winner
    if score_180 > score_0:
        cv2.rectangle(c180_s, (0, 0), (tw - 1, max_h - 1), (0, 255, 0), 3)
    else:
        cv2.rectangle(c0_s, (0, 0), (tw - 1, max_h - 1), (0, 255, 0), 3)

    return np.hstack([ref_s, c0_s, c180_s])


def draw_warp_overlay(reference: np.ndarray, warped: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    """Blend reference and warped image for alignment inspection."""
    def _to_bgr(img):
        return (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if len(img.shape) == 2 else img.copy())

    ref_bgr = _to_bgr(reference)
    warp_bgr = _to_bgr(warped)

    if ref_bgr.shape != warp_bgr.shape:
        warp_bgr = cv2.resize(warp_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))

    return cv2.addWeighted(ref_bgr, alpha, warp_bgr, 1 - alpha, 0)


# Ordered list of debug image names produced by the alignment pipeline.
# Used to assign numbered prefixes when saving.
_DEBUG_ORDER = [
    'keypoints_ref', 'keypoints_cur', 'matches',
    'warp_coarse', 'orientation_comparison', 'orientation_selected',
    'warp_overlay', 'ecc_refined', 'final',
]


def save_debug_set(debug_images: Dict[str, np.ndarray],
                   output_dir: str):
    """Save a dictionary of debug images with numbered prefixes."""
    os.makedirs(output_dir, exist_ok=True)

    idx = 1
    for name in _DEBUG_ORDER:
        if name in debug_images:
            path = os.path.join(output_dir, f"{idx:02d}_{name}.png")
            cv2.imwrite(path, debug_images[name])
            idx += 1

    # Any remaining images not in the predefined order
    for name, img in debug_images.items():
        if name not in _DEBUG_ORDER:
            path = os.path.join(output_dir, f"{idx:02d}_{name}.png")
            cv2.imwrite(path, img)
            idx += 1
