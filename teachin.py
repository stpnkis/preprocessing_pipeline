"""Teach-in workflow for MANTIS preprocessing pipeline.

Builds a product alignment profile from a small set of normal reference images.
The profile is saved to disk and reused during batch preprocessing.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

from . import utils
from .alignment import FeatureAligner
from .config import ProductProfile, load_config
from .debug_viz import draw_keypoints, draw_warp_overlay


def run_teachin(input_dir: str, output_dir: str, config: Dict[str, Any],
                roi: Optional[List[int]] = None):
    """Run teach-in on a set of normal reference images.

    Args:
        input_dir:  Directory containing normal teach-in images.
        output_dir: Directory to save the product profile.
        config:     Pipeline configuration dict.
        roi:        Optional [x, y, w, h] region of interest in the
                    reference frame.  If given, the canonical output will
                    be cropped to this region.

    Returns:
        ProductProfile instance.
    """
    product_id = config.get('product_id', 'unknown')
    print(f"[teach-in] product='{product_id}'")
    print(f"[teach-in] input  = {input_dir}")
    print(f"[teach-in] output = {output_dir}")

    # ------------------------------------------------------------------
    # 1. Load teach-in images
    # ------------------------------------------------------------------
    image_paths = utils.list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"[teach-in] found {len(image_paths)} images")
    images = []
    for p in image_paths:
        images.append(utils.load_image(p))

    # ------------------------------------------------------------------
    # 2. Use first image as initial reference
    # ------------------------------------------------------------------
    aligner = FeatureAligner(config)
    initial_ref = images[0]
    initial_ref_gray = utils.to_gray(initial_ref)
    ref_h, ref_w = initial_ref_gray.shape[:2]
    reference_size = (ref_w, ref_h)              # (width, height)

    print(f"[teach-in] initial ref: {os.path.basename(image_paths[0])} "
          f"({ref_w}x{ref_h})")

    ref_kp, ref_desc = aligner.detect_and_compute(initial_ref_gray)
    print(f"[teach-in] initial ref features: {len(ref_kp)}")

    if len(ref_kp) < 50:
        print("[teach-in] WARNING: very few features on initial reference — "
              "consider using AKAZE with a lower threshold or a sharper image")

    # ------------------------------------------------------------------
    # 3. Debug directory
    # ------------------------------------------------------------------
    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    utils.save_image(
        os.path.join(debug_dir, '00_initial_reference.png'),
        initial_ref,
    )
    utils.save_image(
        os.path.join(debug_dir, '01_initial_reference_keypoints.png'),
        draw_keypoints(initial_ref_gray, ref_kp),
    )

    # ------------------------------------------------------------------
    # 4. Align all other teach-in images to the initial reference
    # ------------------------------------------------------------------
    aligned_images = [initial_ref.copy()]          # the reference itself
    alignment_stats: List[Dict[str, Any]] = []

    for i, (img, path) in enumerate(zip(images[1:], image_paths[1:]),
                                     start=1):
        name = os.path.basename(path)
        print(f"[teach-in]   aligning {i}/{len(images)-1}: {name} ...",
              end=" ", flush=True)

        result = aligner.align(
            img, initial_ref_gray, ref_kp, ref_desc,
            canonical_size=reference_size, roi=roi, save_debug=False,
        )

        if result.success:
            aligned_images.append(result.aligned_image)
            alignment_stats.append({
                'image': name,
                'inliers': result.num_inliers,
                'inlier_ratio': result.inlier_ratio,
                'similarity': result.similarity_score,
                'ecc_score': result.ecc_score,
                'flipped': result.orientation_flipped,
            })
            utils.save_image(
                os.path.join(debug_dir, f'aligned_teachin_{i:03d}.png'),
                result.aligned_image,
            )
            print(f"OK  inliers={result.num_inliers}  "
                  f"sim={result.similarity_score:.3f}")
        else:
            print(f"FAIL ({result.status}) — excluded from median")

    # ------------------------------------------------------------------
    # 5. Build median / consensus reference
    # ------------------------------------------------------------------
    n_aligned = len(aligned_images)
    if n_aligned >= 3:
        print(f"[teach-in] building median reference from {n_aligned} images")
        target_h, target_w = aligned_images[0].shape[:2]
        stack = []
        for img in aligned_images:
            if img.shape[:2] == (target_h, target_w):
                stack.append(img)
            else:
                stack.append(cv2.resize(img, (target_w, target_h)))
        reference = np.median(
            np.stack(stack, axis=0), axis=0,
        ).astype(np.uint8)
    elif n_aligned == 2:
        print("[teach-in] averaging 2 aligned images for reference")
        reference = (
            (aligned_images[0].astype(np.float64)
             + aligned_images[1].astype(np.float64)) / 2
        ).astype(np.uint8)
    else:
        print("[teach-in] using single image as reference")
        reference = aligned_images[0]

    # ------------------------------------------------------------------
    # 6. Re-extract features on final reference
    # ------------------------------------------------------------------
    reference_gray = utils.to_gray(reference)
    final_kp, final_desc = aligner.detect_and_compute(reference_gray)
    print(f"[teach-in] final reference features: {len(final_kp)}")

    # ------------------------------------------------------------------
    # 7. Validate stability across teach-in set
    # ------------------------------------------------------------------
    if alignment_stats:
        _validate_stability(alignment_stats, config)

    # ------------------------------------------------------------------
    # 8. Canonical output size
    # ------------------------------------------------------------------
    if roi is not None:
        canonical_size = (roi[2], roi[3])          # (width, height)
    else:
        canonical_size = reference_size

    # ------------------------------------------------------------------
    # 9. Compute teach-in summary stats
    # ------------------------------------------------------------------
    teachin_stats = _compute_teachin_stats(alignment_stats, len(images))

    # ------------------------------------------------------------------
    # 10. Persist profile
    # ------------------------------------------------------------------
    profile = ProductProfile(output_dir)
    profile.save(
        config=config,
        reference_size=reference_size,
        canonical_size=canonical_size,
        roi=roi,
        teachin_stats=teachin_stats,
    )

    utils.save_image(profile.reference_image_path, reference)
    utils.save_features(profile.features_path, final_kp, final_desc)

    # Debug: final reference + keypoints + overlay with initial
    utils.save_image(
        os.path.join(debug_dir, '02_final_reference.png'), reference,
    )
    utils.save_image(
        os.path.join(debug_dir, '03_final_reference_keypoints.png'),
        draw_keypoints(reference_gray, final_kp),
    )
    utils.save_image(
        os.path.join(debug_dir, '04_overlay_initial_vs_final.png'),
        draw_warp_overlay(initial_ref_gray, reference_gray),
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n[teach-in] DONE")
    print(f"  Profile:        {profile.profile_path}")
    print(f"  Reference:      {profile.reference_image_path}")
    print(f"  Canonical size: {canonical_size[0]}x{canonical_size[1]}")
    if teachin_stats.get('mean_similarity'):
        print(f"  Mean similarity: {teachin_stats['mean_similarity']}")
        print(f"  Mean inliers:    {teachin_stats['mean_inliers']}")

    return profile


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _compute_teachin_stats(stats: list,
                           total_images: int) -> Dict[str, Any]:
    """Compute summary statistics from teach-in alignment results."""
    if not stats:
        return {'num_teachin_images': total_images, 'num_aligned': 1}

    inliers = [s['inliers'] for s in stats]
    sims = [s['similarity'] for s in stats]

    return {
        'num_teachin_images': total_images,
        'num_aligned': len(stats) + 1,            # +1 for the reference
        'mean_inliers': round(sum(inliers) / len(inliers), 1),
        'min_inliers': min(inliers),
        'max_inliers': max(inliers),
        'mean_similarity': round(sum(sims) / len(sims), 4),
        'min_similarity': round(min(sims), 4),
        'max_similarity': round(max(sims), 4),
    }


def _validate_stability(stats: list, config: dict):
    """Warn if teach-in alignment quality is inconsistent."""
    sims = [s['similarity'] for s in stats]
    inliers = [s['inliers'] for s in stats]
    ratios = [s['inlier_ratio'] for s in stats]

    gates = config.get('quality_gates', {})
    min_sim = gates.get('min_similarity_score', 0.3)

    if min(sims) < min_sim:
        print(f"[teach-in] WARNING: worst teach-in similarity "
              f"({min(sims):.3f}) is below gate ({min_sim:.3f})")

    if max(sims) - min(sims) > 0.3:
        print(f"[teach-in] WARNING: large similarity spread across "
              f"teach-in set ({min(sims):.3f} – {max(sims):.3f}). "
              f"Check image quality / pose variability.")

    if min(inliers) < gates.get('min_inlier_count', 10):
        print(f"[teach-in] WARNING: min inlier count ({min(inliers)}) "
              f"below threshold — some teach-in images are marginal")
