"""Interactive region selection tools using OpenCV for MANTIS pipeline.

Provides lightweight GUI helpers for defining:
  - rectangular ROIs (canonical crop, similarity ROI)
  - polygon masks (object mask)
  - combined region preview

These work on any system with an OpenCV-capable display (X11, Wayland, etc.).
If no display is available, regions can be set by editing profile.yaml directly.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


def select_rectangle(image: np.ndarray,
                     title: str = "Select region") -> Optional[List[int]]:
    """Interactive rectangle selection using OpenCV selectROI.

    Drag to draw a rectangle, then press ENTER or SPACE to confirm.
    Press C or ESC to cancel.

    Returns [x, y, w, h] in original image coordinates, or None.
    """
    display, scale = _prepare_display(image)

    try:
        r = cv2.selectROI(title, display, fromCenter=False,
                          showCrosshair=True)
    except cv2.error:
        print("[regions] OpenCV GUI not available — edit profile.yaml manually")
        return None
    finally:
        cv2.destroyWindow(title)

    if r[2] == 0 or r[3] == 0:
        return None

    # Scale back to original image coordinates
    return [int(r[0] / scale), int(r[1] / scale),
            int(r[2] / scale), int(r[3] / scale)]


def select_polygon(image: np.ndarray,
                   title: str = "Draw polygon") -> Optional[List[List[int]]]:
    """Interactive polygon drawing via mouse clicks.

    LEFT-click to add a vertex.
    RIGHT-click or press ENTER/SPACE to close the polygon.
    Press ESC to cancel.

    Returns list of [x, y] vertices in original coordinates, or None.
    """
    display, scale = _prepare_display(image)
    base = display.copy()
    points: List[List[int]] = []
    done = [False]

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            _redraw(base, points, title)
        elif event == cv2.EVENT_RBUTTONDOWN:
            done[0] = True

    try:
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, _on_mouse)
        cv2.imshow(title, display)

        while not done[0]:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:           # ESC → cancel
                cv2.destroyWindow(title)
                return None
            if key in (13, 32):     # ENTER / SPACE → confirm
                break
    except cv2.error:
        print("[regions] OpenCV GUI not available — edit profile.yaml manually")
        return None
    finally:
        cv2.destroyWindow(title)

    if len(points) < 3:
        return None

    # Scale back to original coordinates
    return [[int(p[0] / scale), int(p[1] / scale)] for p in points]


def preview_regions(image: np.ndarray,
                    canonical_crop: Optional[List[int]] = None,
                    object_mask: Optional[List[List[int]]] = None,
                    similarity_roi: Optional[List[int]] = None,
                    title: str = "Region preview — any key to close"):
    """Display image with all defined regions overlaid.

    Green  = canonical_crop
    Cyan   = similarity_roi
    Magenta = object_mask polygon
    """
    display, scale = _prepare_display(image)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if canonical_crop is not None:
        x, y, w, h = [int(v * scale) for v in canonical_crop]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, "canonical_crop", (x, y - 6),
                    font, 0.5, (0, 255, 0), 1)

    if similarity_roi is not None:
        x, y, w, h = [int(v * scale) for v in similarity_roi]
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(display, "similarity_roi", (x, y - 6),
                    font, 0.5, (255, 255, 0), 1)

    if object_mask is not None and len(object_mask) >= 3:
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)]
                        for p in object_mask], dtype=np.int32)
        cv2.polylines(display, [pts], True, (255, 0, 255), 2)
        cv2.putText(display, "object_mask", (pts[0][0], pts[0][1] - 6),
                    font, 0.5, (255, 0, 255), 1)

    try:
        cv2.imshow(title, display)
        cv2.waitKey(0)
    except cv2.error:
        pass
    finally:
        cv2.destroyAllWindows()


def save_region_visualization(image: np.ndarray,
                              regions: dict,
                              output_path: str):
    """Render and save region overlay to a file (no GUI needed)."""
    vis = image.copy() if len(image.shape) == 3 else \
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    crop = regions.get('canonical_crop')
    if crop:
        x, y, w, h = crop
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(vis, "canonical_crop", (x + 5, y + 25),
                    font, 0.7, (0, 255, 0), 2)

    sim = regions.get('similarity_roi')
    if sim:
        x, y, w, h = sim
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(vis, "similarity_roi", (x + 5, y + 25),
                    font, 0.7, (255, 255, 0), 2)

    mask_poly = regions.get('object_mask')
    if mask_poly and len(mask_poly) >= 3:
        pts = np.array(mask_poly, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (255, 0, 255), 2)
        cv2.putText(vis, "object_mask", (pts[0][0] + 5, pts[0][1] - 10),
                    font, 0.7, (255, 0, 255), 2)

    cv2.imwrite(output_path, vis)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _prepare_display(image: np.ndarray,
                     max_dim: int = 1200) -> Tuple[np.ndarray, float]:
    """Scale image for comfortable on-screen display."""
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        display = image.copy()

    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    return display, scale


def _redraw(base: np.ndarray, points: list, title: str):
    """Redraw polygon in-progress."""
    vis = base.copy()
    if points:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=False,
                      color=(0, 255, 0), thickness=2)
        for p in points:
            cv2.circle(vis, (p[0], p[1]), 4, (0, 0, 255), -1)
    cv2.imshow(title, vis)
