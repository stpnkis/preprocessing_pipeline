"""Batch preprocessing workflow for MANTIS pipeline.

Processes folders of images using a saved product profile, producing
aligned canonical outputs with metadata and optional debug visualisations.
"""

import os
import shutil
from typing import Dict, Any, List, Optional

from . import utils
from .alignment import FeatureAligner
from .config import ProductProfile
from .debug_viz import (draw_keypoints, draw_warp_overlay,
                        save_debug_set)
from .quality import (save_alignment_metadata, generate_summary_csv,
                      print_summary)


def run_preprocess(profile_dir: str, input_dir: str, output_dir: str,
                   save_debug: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Batch-preprocess images using a saved product profile.

    Args:
        profile_dir: Directory containing the product profile.
        input_dir:   Directory of raw images to process.
        output_dir:  Root output directory.
        save_debug:  Override profile's save_debug setting.

    Returns:
        List of per-image summary records.
    """
    # ------------------------------------------------------------------
    # 1. Load profile and reference
    # ------------------------------------------------------------------
    profile = ProductProfile(profile_dir)
    profile_data = profile.load()
    config = profile.get_config()

    if save_debug is not None:
        config['output']['save_debug'] = save_debug
    do_debug = config['output'].get('save_debug', True)

    product_id = profile_data.get('product_id', 'unknown')
    reference_size = tuple(profile_data['reference_size'])     # (w, h)
    canonical_size = tuple(
        profile_data.get('canonical_size', reference_size),
    )
    sim_roi = profile.get_similarity_roi()
    canonical_crop = profile.get_canonical_crop()

    print(f"[preprocess] product='{product_id}'")
    print(f"[preprocess] profile = {profile.profile_path}")
    print(f"[preprocess] input   = {input_dir}")
    print(f"[preprocess] output  = {output_dir}")
    print(f"[preprocess] reference {reference_size[0]}x{reference_size[1]}  "
          f"canonical {canonical_size[0]}x{canonical_size[1]}  "
          f"crop={'yes' if canonical_crop else 'no'}  "
          f"debug={do_debug}")

    reference = utils.load_image(profile.reference_image_path)
    reference_gray = utils.to_gray(reference)
    ref_kp, ref_desc = utils.load_features(profile.features_path)
    print(f"[preprocess] reference features: {len(ref_kp)}")

    aligner = FeatureAligner(config)

    # ------------------------------------------------------------------
    # 2. Prepare output directories
    # ------------------------------------------------------------------
    aligned_dir = os.path.join(output_dir, 'aligned')
    metadata_dir = os.path.join(output_dir, 'metadata')
    debug_dir = os.path.join(output_dir, 'debug')
    failed_dir = os.path.join(output_dir, 'failed')
    reports_dir = os.path.join(output_dir, 'reports')

    for d in [aligned_dir, metadata_dir, failed_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
    if do_debug:
        os.makedirs(debug_dir, exist_ok=True)

    # Pre-render reference keypoints for debug sets
    ref_kp_vis = draw_keypoints(reference_gray, ref_kp) if do_debug else None

    # ------------------------------------------------------------------
    # 3. Process images
    # ------------------------------------------------------------------
    image_paths = utils.list_images(input_dir)
    if not image_paths:
        print("[preprocess] no images found in input directory")
        return []

    print(f"[preprocess] processing {len(image_paths)} images ...\n")
    records: List[Dict[str, Any]] = []

    for idx, img_path in enumerate(image_paths):
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        print(f"  [{idx+1}/{len(image_paths)}] {name} ... ",
              end="", flush=True)

        # Load
        try:
            image = utils.load_image(img_path)
        except IOError as e:
            print(f"LOAD ERROR: {e}")
            records.append(_empty_record(name, 'load_error'))
            continue

        # Align
        result = aligner.align(
            image, reference_gray, ref_kp, ref_desc,
            canonical_size=reference_size,
            similarity_roi=sim_roi,
            canonical_crop=canonical_crop,
            save_debug=do_debug,
        )

        # Build summary record
        meta = result.metadata
        record = {
            'image': name,
            'status': result.status,
            'num_features': meta.get('num_features_detected', 0),
            'num_matches': meta.get('num_matches', 0),
            'num_inliers': result.num_inliers,
            'inlier_ratio': result.inlier_ratio,
            'similarity': result.similarity_score,
            'ecc_score': result.ecc_score,
            'orientation_flipped': result.orientation_flipped,
        }
        records.append(record)

        # Save outputs
        if result.success:
            utils.save_image(
                os.path.join(aligned_dir, f"{stem}.png"),
                result.aligned_image,
            )
            print(f"OK  sim={result.similarity_score:.3f}  "
                  f"inliers={result.num_inliers}")
        else:
            shutil.copy2(img_path, os.path.join(failed_dir, name))
            print(f"FAILED: {result.status}")

        # Per-image metadata
        save_alignment_metadata(
            os.path.join(metadata_dir, f"{stem}.json"),
            meta, name, result.status,
        )

        # Debug images
        if do_debug and result.debug_images:
            debug_set = result.debug_images.copy()
            if ref_kp_vis is not None:
                debug_set['keypoints_ref'] = ref_kp_vis

            # Overlay: warped vs reference
            warped_for_overlay = debug_set.get(
                'orientation_selected',
                debug_set.get('warp_coarse'),
            )
            if warped_for_overlay is not None:
                debug_set['warp_overlay'] = draw_warp_overlay(
                    reference_gray, utils.to_gray(warped_for_overlay),
                )
            save_debug_set(debug_set, os.path.join(debug_dir, stem))

    # ------------------------------------------------------------------
    # 4. Summary report
    # ------------------------------------------------------------------
    csv_path = os.path.join(reports_dir, 'summary.csv')
    generate_summary_csv(records, csv_path)
    print_summary(records)
    print(f"[preprocess] summary → {csv_path}")

    return records


def _empty_record(name: str, status: str) -> Dict[str, Any]:
    return {
        'image': name, 'status': status,
        'num_features': 0, 'num_matches': 0, 'num_inliers': 0,
        'inlier_ratio': 0.0, 'similarity': 0.0, 'ecc_score': 0.0,
        'orientation_flipped': False,
    }
