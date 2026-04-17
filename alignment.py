"""Core alignment engine for MANTIS preprocessing pipeline.

Implements feature-based coarse registration (AKAZE/ORB + RANSAC homography),
180-degree orientation resolution, and optional ECC refinement for planar
PCB-like objects.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from . import utils
from .debug_viz import draw_keypoints, draw_matches, draw_orientation_comparison


@dataclass
class AlignmentResult:
    """Result of aligning a single image."""
    success: bool
    status: str  # 'ok', 'insufficient_matches', 'homography_failed', etc.
    aligned_image: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    ecc_score: float = 0.0
    orientation_flipped: bool = False
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureAligner:
    """Feature-based image alignment for planar objects.

    Pipeline per image:
        1. Detect features (AKAZE or ORB)
        2. Match descriptors (KNN + ratio test)
        3. Estimate homography (RANSAC)
        4. Warp to reference frame
        5. Resolve 180° ambiguity via NCC
        6. Optional ECC refinement
        7. Crop to canonical output
    """

    def __init__(self, config: dict):
        self.config = config
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()

    # ------------------------------------------------------------------
    # Detector / matcher construction
    # ------------------------------------------------------------------

    def _create_detector(self):
        feat_type = self.config.get('feature_detector', 'akaze')
        if feat_type == 'akaze':
            params = self.config.get('feature_params', {}).get('akaze', {})
            return cv2.AKAZE_create(
                descriptor_type=params.get('descriptor_type', cv2.AKAZE_DESCRIPTOR_MLDB),
                threshold=params.get('threshold', 0.0003),
            )
        elif feat_type == 'orb':
            params = self.config.get('feature_params', {}).get('orb', {})
            return cv2.ORB_create(
                nfeatures=params.get('nfeatures', 5000),
                scaleFactor=params.get('scale_factor', 1.2),
                nlevels=params.get('nlevels', 8),
            )
        else:
            raise ValueError(f"Unknown feature detector: {feat_type}")

    def _create_matcher(self):
        feat_type = self.config.get('feature_detector', 'akaze')
        if feat_type == 'akaze':
            desc_type = (self.config.get('feature_params', {})
                         .get('akaze', {})
                         .get('descriptor_type', cv2.AKAZE_DESCRIPTOR_MLDB))
            if desc_type in (cv2.AKAZE_DESCRIPTOR_MLDB,
                             cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT):
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif feat_type == 'orb':
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # ------------------------------------------------------------------
    # Feature detection and matching
    # ------------------------------------------------------------------

    def detect_and_compute(self, gray: np.ndarray,
                           mask: Optional[np.ndarray] = None):
        """Detect keypoints and compute descriptors on a grayscale image.

        Args:
            gray: Grayscale input image.
            mask: Optional uint8 mask (255 = detect here, 0 = ignore).
        """
        kps, descs = self.detector.detectAndCompute(gray, mask)
        return list(kps) if kps is not None else [], descs

    def match_features(self, desc_ref: np.ndarray,
                       desc_cur: np.ndarray) -> List[cv2.DMatch]:
        """Match descriptors using KNN + Lowe's ratio test.

        desc_ref: reference descriptors (train)
        desc_cur: current image descriptors (query)

        Returns good matches where queryIdx → desc_cur, trainIdx → desc_ref.
        """
        ratio_thresh = self.config.get('matching', {}).get('ratio_threshold', 0.75)

        if desc_ref is None or desc_cur is None:
            return []
        if len(desc_ref) < 2 or len(desc_cur) < 2:
            return []

        raw_matches = self.matcher.knnMatch(desc_cur, desc_ref, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)
        return good

    # ------------------------------------------------------------------
    # Homography estimation
    # ------------------------------------------------------------------

    def estimate_homography(self, kp_ref: list, kp_cur: list,
                            matches: List[cv2.DMatch]):
        """Estimate homography via RANSAC.

        Returns (H, inlier_mask, num_inliers, inlier_ratio).
        H maps from current image coords to reference coords.
        """
        cfg = self.config.get('ransac', {})
        reproj = cfg.get('reproj_threshold', 5.0)
        max_iters = cfg.get('max_iters', 5000)
        confidence = cfg.get('confidence', 0.999)

        if len(matches) < 4:
            return None, None, 0, 0.0

        pts_cur = np.float32(
            [kp_cur[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        pts_ref = np.float32(
            [kp_ref[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            pts_cur, pts_ref, cv2.RANSAC,
            ransacReprojThreshold=reproj,
            maxIters=max_iters,
            confidence=confidence,
        )

        if H is None or mask is None:
            return None, None, 0, 0.0

        inlier_mask = mask.ravel().astype(bool)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / len(matches)

        return H, inlier_mask, num_inliers, inlier_ratio

    def _is_homography_reasonable(self, H: np.ndarray,
                                  img_shape: Tuple[int, ...],
                                  target_size: Tuple[int, int]) -> bool:
        """Sanity-check a homography by warping image corners.

        target_size is (width, height).
        """
        gates = self.config.get('quality_gates', {})
        max_ratio = gates.get('max_warp_area_ratio', 10.0)
        min_ratio = gates.get('min_warp_area_ratio', 0.1)

        h, w = img_shape[:2]
        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]
        ).reshape(-1, 1, 2)

        try:
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        except cv2.error:
            return False

        # Convexity check
        hull = cv2.convexHull(warped.astype(np.float32))
        if len(hull) < 4:
            return False

        # Area ratio check
        area_warped = abs(cv2.contourArea(warped.astype(np.float32)))
        tw, th = target_size
        area_target = tw * th
        if area_target == 0:
            return False
        ratio = area_warped / area_target
        if ratio < min_ratio or ratio > max_ratio:
            return False

        return True

    # ------------------------------------------------------------------
    # Warp and orientation
    # ------------------------------------------------------------------

    def warp_image(self, image: np.ndarray, H: np.ndarray,
                   size: Tuple[int, int]) -> np.ndarray:
        """Warp image using homography.  size = (width, height)."""
        return cv2.warpPerspective(
            image, H, size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def check_orientation(self, warped_gray: np.ndarray,
                          reference_gray: np.ndarray,
                          roi: Optional[list] = None):
        """Compare 0° and 180° orientations via NCC.

        Returns (needs_flip, score_0, score_180, warped_180_gray).
        """
        if not self.config.get('orientation', {}).get('check_180', True):
            return False, 1.0, 0.0, None

        h, w = warped_gray.shape[:2]
        center = (w / 2.0, h / 2.0)

        score_0 = self._similarity(warped_gray, reference_gray, roi)

        M_180 = cv2.getRotationMatrix2D(center, 180, 1.0)
        warped_180 = cv2.warpAffine(warped_gray, M_180, (w, h))
        score_180 = self._similarity(warped_180, reference_gray, roi)

        needs_flip = score_180 > score_0
        return needs_flip, score_0, score_180, warped_180

    def _similarity(self, img1: np.ndarray, img2: np.ndarray,
                    roi: Optional[list] = None) -> float:
        """Normalized cross-correlation between two grayscale images."""
        if roi is not None:
            x, y, rw, rh = roi
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            # Clamp ROI to image bounds
            x1e = min(x + rw, w1, w2)
            y1e = min(y + rh, h1, h2)
            x = max(0, x)
            y = max(0, y)
            if x1e <= x or y1e <= y:
                return 0.0
            img1 = img1[y:y1e, x:x1e]
            img2 = img2[y:y1e, x:x1e]

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        a = img1.astype(np.float64)
        b = img2.astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()

        num = np.sum(a * b)
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        if denom < 1e-10:
            return 0.0
        return float(num / denom)

    # ------------------------------------------------------------------
    # ECC refinement
    # ------------------------------------------------------------------

    def refine_ecc(self, warped_gray: np.ndarray,
                   reference_gray: np.ndarray,
                   roi: Optional[list] = None):
        """ECC refinement on full-size images.

        Returns (warp_matrix, ecc_score) or (None, 0.0) on failure.
        The warp_matrix is in full-image coordinates.
        """
        ecfg = self.config.get('ecc', {})
        if not ecfg.get('enabled', True):
            return None, 0.0

        max_iter = ecfg.get('max_iterations', 200)
        epsilon = ecfg.get('epsilon', 1e-5)
        gauss = ecfg.get('gaussian_filter_size', 5)

        motion_map = {
            'euclidean': cv2.MOTION_EUCLIDEAN,
            'affine': cv2.MOTION_AFFINE,
            'translation': cv2.MOTION_TRANSLATION,
            'homography': cv2.MOTION_HOMOGRAPHY,
        }
        motion_type = motion_map.get(
            ecfg.get('motion_type', 'euclidean'),
            cv2.MOTION_EUCLIDEAN,
        )

        if motion_type == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iter, epsilon,
        )

        # Build mask if ROI specified (full-image coords)
        mask = None
        if roi is not None:
            mask = np.zeros(reference_gray.shape[:2], dtype=np.uint8)
            x, y, rw, rh = roi
            rh_img, rw_img = reference_gray.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + rw, rw_img)
            y2 = min(y + rh, rh_img)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        try:
            cc, warp_matrix = cv2.findTransformECC(
                reference_gray, warped_gray, warp_matrix,
                motion_type, criteria, mask, gauss,
            )
            return warp_matrix, float(cc)
        except cv2.error:
            return None, 0.0

    def apply_ecc_warp(self, image: np.ndarray, warp_matrix: np.ndarray,
                       size: Tuple[int, int]) -> np.ndarray:
        """Apply ECC warp matrix to an image.  size = (width, height)."""
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        if warp_matrix.shape[0] == 3:
            return cv2.warpPerspective(image, warp_matrix, size, flags=flags)
        else:
            return cv2.warpAffine(image, warp_matrix, size, flags=flags)

    # ------------------------------------------------------------------
    # Full alignment pipeline
    # ------------------------------------------------------------------

    def align(self, image: np.ndarray,
              reference_gray: np.ndarray,
              ref_keypoints: list,
              ref_descriptors: np.ndarray,
              canonical_size: Tuple[int, int],
              roi: Optional[list] = None,
              similarity_roi: Optional[list] = None,
              canonical_crop: Optional[list] = None,
              save_debug: bool = False) -> AlignmentResult:
        """Align a single image to the canonical reference.

        Args:
            image:           Input BGR image.
            reference_gray:  Reference grayscale image.
            ref_keypoints:   Reference keypoints.
            ref_descriptors: Reference descriptors.
            canonical_size:  (width, height) — warp target size
                             (normally the reference image size).
            roi:             Legacy [x, y, w, h] used for both similarity
                             and crop when the new params are not set.
            similarity_roi:  [x, y, w, h] for orientation check, ECC,
                             and quality measurement.  Falls back to
                             canonical_crop, then roi.
            canonical_crop:  [x, y, w, h] for final output crop.
                             Falls back to roi.
            save_debug:      Whether to produce debug visualisations.

        Returns:
            AlignmentResult with aligned image (or None on failure).
        """
        # Resolve effective ROIs for backward compatibility
        eff_sim_roi = (similarity_roi if similarity_roi is not None
                       else (canonical_crop if canonical_crop is not None
                             else roi))
        eff_crop = canonical_crop if canonical_crop is not None else roi

        debug = {}
        meta: Dict[str, Any] = {}
        gates = self.config.get('quality_gates', {})

        # --- grayscale -------------------------------------------------
        gray = utils.to_gray(image)

        # --- 1. detect features ----------------------------------------
        kp_cur, desc_cur = self.detect_and_compute(gray)
        meta['num_features_detected'] = len(kp_cur)

        if save_debug:
            debug['keypoints_cur'] = draw_keypoints(gray, kp_cur)

        # --- 2. match --------------------------------------------------
        matches = self.match_features(ref_descriptors, desc_cur)
        meta['num_matches'] = len(matches)

        min_matches = self.config.get('matching', {}).get('min_matches', 15)
        if len(matches) < min_matches:
            return AlignmentResult(
                success=False, status='insufficient_matches',
                metadata=meta, debug_images=debug,
            )

        # --- 3. homography ---------------------------------------------
        H, inlier_mask, n_inliers, inlier_ratio = self.estimate_homography(
            ref_keypoints, kp_cur, matches,
        )
        meta['num_inliers'] = n_inliers
        meta['inlier_ratio'] = round(inlier_ratio, 4)

        if save_debug:
            # draw only inlier matches
            inlier_matches = (
                [m for m, ok in zip(matches, inlier_mask) if ok]
                if inlier_mask is not None else matches
            )
            debug['matches'] = draw_matches(
                gray, kp_cur, reference_gray, ref_keypoints,
                inlier_matches, max_display=200,
            )

        if H is None:
            return AlignmentResult(
                success=False, status='homography_failed',
                metadata=meta, debug_images=debug,
            )

        # Quality gate: inlier count / ratio
        if n_inliers < gates.get('min_inlier_count', 10):
            return AlignmentResult(
                success=False, status='low_inlier_count',
                num_inliers=n_inliers, inlier_ratio=inlier_ratio,
                metadata=meta, debug_images=debug,
            )
        if inlier_ratio < gates.get('min_inlier_ratio', 0.15):
            return AlignmentResult(
                success=False, status='low_inlier_ratio',
                num_inliers=n_inliers, inlier_ratio=inlier_ratio,
                metadata=meta, debug_images=debug,
            )

        # Geometric sanity check
        if not self._is_homography_reasonable(H, gray.shape, canonical_size):
            return AlignmentResult(
                success=False, status='unreasonable_homography',
                num_inliers=n_inliers, inlier_ratio=inlier_ratio,
                metadata=meta, debug_images=debug,
            )

        # --- 4. coarse warp -------------------------------------------
        warped_coarse = self.warp_image(image, H, canonical_size)
        warped_coarse_gray = utils.to_gray(warped_coarse)

        if save_debug:
            debug['warp_coarse'] = warped_coarse.copy()

        # --- 5. orientation check (180°) -------------------------------
        needs_flip, score_0, score_180, warped_180_gray = \
            self.check_orientation(warped_coarse_gray, reference_gray,
                                  eff_sim_roi)

        meta['orientation_score_0'] = round(score_0, 4)
        meta['orientation_score_180'] = round(score_180, 4)
        meta['orientation_flipped'] = needs_flip

        if save_debug:
            debug['orientation_comparison'] = draw_orientation_comparison(
                reference_gray, warped_coarse_gray,
                warped_180_gray if warped_180_gray is not None
                else warped_coarse_gray,
                score_0, score_180,
            )

        if needs_flip:
            h, w = warped_coarse.shape[:2]
            center = (w / 2.0, h / 2.0)
            M_180 = cv2.getRotationMatrix2D(center, 180, 1.0)
            warped_coarse = cv2.warpAffine(warped_coarse, M_180, (w, h))
            warped_coarse_gray = utils.to_gray(warped_coarse)

        if save_debug:
            debug['orientation_selected'] = warped_coarse.copy()

        # Similarity after coarse warp + orientation
        similarity = self._similarity(warped_coarse_gray, reference_gray,
                                      eff_sim_roi)
        meta['similarity_after_warp'] = round(similarity, 4)

        if similarity < gates.get('min_similarity_score', 0.3):
            return AlignmentResult(
                success=False, status='low_similarity',
                aligned_image=warped_coarse,
                similarity_score=similarity,
                num_inliers=n_inliers, inlier_ratio=inlier_ratio,
                orientation_flipped=needs_flip,
                metadata=meta, debug_images=debug,
            )

        # --- 6. ECC refinement ----------------------------------------
        ecc_matrix, ecc_score = self.refine_ecc(
            warped_coarse_gray, reference_gray, eff_sim_roi,
        )
        meta['ecc_score'] = round(ecc_score, 4)

        final = warped_coarse
        final_similarity = similarity

        if ecc_matrix is not None:
            refined = self.apply_ecc_warp(warped_coarse, ecc_matrix,
                                          canonical_size)
            refined_gray = utils.to_gray(refined)
            sim_ecc = self._similarity(refined_gray, reference_gray,
                                        eff_sim_roi)
            meta['similarity_after_ecc'] = round(sim_ecc, 4)

            if sim_ecc >= similarity:
                final = refined
                final_similarity = sim_ecc
            else:
                meta['ecc_rejected'] = True
        else:
            meta['ecc_converged'] = False

        if save_debug:
            debug['ecc_refined'] = final.copy()

        # --- 7. Canonical crop -----------------------------------------
        if eff_crop is not None:
            x, y, rw, rh = eff_crop
            fh, fw = final.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + rw, fw)
            y2 = min(y + rh, fh)
            if x2 > x1 and y2 > y1:
                final_crop = final[y1:y2, x1:x2].copy()
            else:
                final_crop = final
        else:
            final_crop = final

        if save_debug:
            debug['final'] = final_crop.copy()

        return AlignmentResult(
            success=True, status='ok',
            aligned_image=final_crop,
            similarity_score=round(final_similarity, 4),
            num_inliers=n_inliers,
            inlier_ratio=round(inlier_ratio, 4),
            ecc_score=round(ecc_score, 4),
            orientation_flipped=needs_flip,
            metadata=meta,
            debug_images=debug,
        )
