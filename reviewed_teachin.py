"""Reviewed teach-in workflow for MANTIS preprocessing pipeline.

Provides a controlled, multi-step teach-in process that lets the operator
inspect and tune every decision before committing to a final profile.

Steps:
    1. init     — select teach-in images from the real dataset, set up
                  a draft profile with a primary reference
    2. regions  — define canonical crop / object mask / similarity ROI
                  (interactively or by editing profile.yaml)
    3. preview  — align teach-in images using current settings and
                  inspect debug visualisations
    4. finalize — build a median-consensus reference from the aligned
                  teach-in images and lock the profile for production use
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

from . import utils
from .alignment import FeatureAligner
from .config import ProductProfile, load_config, _deep_copy, _CONFIG_KEYS
from .debug_viz import (draw_keypoints, draw_warp_overlay, save_debug_set)


# ==================================================================
# Step 1: init
# ==================================================================

def init_profile(dataset_path: str, profile_dir: str,
                 config: Dict[str, Any],
                 num_images: int = 10,
                 image_list: Optional[List[str]] = None,
                 reference_image: Optional[str] = None):
    """Initialise a reviewed teach-in profile.

    Scans *dataset_path* for images, selects a teach-in subset,
    chooses a primary reference, extracts features, and writes a
    draft profile to *profile_dir*.

    Args:
        dataset_path:    Directory with normal images.
        profile_dir:     Where to create the profile.
        config:          Base alignment configuration dict.
        num_images:      How many images to auto-select (ignored when
                         *image_list* is provided).
        image_list:      Explicit list of image **filenames** to use.
        reference_image: Filename of the desired primary reference.

    Returns:
        The created :class:`ProductProfile`.
    """
    print(f"[init] dataset = {dataset_path}")
    print(f"[init] profile = {profile_dir}")

    # ---- scan dataset ------------------------------------------------
    all_paths = utils.list_images(dataset_path)
    if not all_paths:
        raise FileNotFoundError(f"No images found in {dataset_path}")
    print(f"[init] {len(all_paths)} images in dataset")

    name_to_path = {os.path.basename(p): p for p in all_paths}
    all_names = list(name_to_path.keys())

    # ---- select teach-in images --------------------------------------
    if image_list:
        selected = []
        for name in image_list:
            if name in name_to_path:
                selected.append(name)
            else:
                print(f"[init] WARNING: '{name}' not found — skipped")
        if not selected:
            raise ValueError("None of the specified images were found")
    else:
        n = min(num_images, len(all_paths))
        step = max(1, len(all_paths) // n)
        selected = [all_names[i]
                    for i in range(0, len(all_paths), step)][:n]

    print(f"[init] selected {len(selected)} teach-in images")

    # ---- choose primary reference ------------------------------------
    if reference_image:
        if reference_image not in name_to_path:
            raise ValueError(
                f"Reference '{reference_image}' not found in dataset")
        ref_name = reference_image
    else:
        ref_name = selected[len(selected) // 2]

    if ref_name not in selected:
        selected.insert(0, ref_name)

    print(f"[init] primary reference: {ref_name}")

    # ---- load reference and extract features -------------------------
    ref_img = utils.load_image(name_to_path[ref_name])
    ref_gray = utils.to_gray(ref_img)
    h, w = ref_gray.shape[:2]
    reference_size = [w, h]

    aligner = FeatureAligner(config)
    ref_kp, ref_desc = aligner.detect_and_compute(ref_gray)
    print(f"[init] reference size: {w}x{h}  features: {len(ref_kp)}")

    if len(ref_kp) < 50:
        print("[init] WARNING: very few features — consider a different "
              "reference or lowing the detector threshold")

    # ---- build profile -----------------------------------------------
    profile = ProductProfile(profile_dir)
    profile.data = {
        'product_id': config.get('product_id', 'pcb1'),
        'teachin_mode': 'reviewed',
        'finalized': False,
        'dataset_source': os.path.abspath(dataset_path),
        'teachin_images': selected,
        'primary_reference': ref_name,
        'reference_image': 'reference.png',
        'reference_features': 'reference_features.npz',
        'reference_size': reference_size,
        'canonical_size': reference_size,
        'regions': {
            'canonical_crop': None,
            'object_mask': None,
            'similarity_roi': None,
        },
    }
    for key in _CONFIG_KEYS:
        if key in config:
            profile.data[key] = config[key]

    profile.save_data()

    # ---- persist reference artefacts ---------------------------------
    utils.save_image(profile.reference_image_path, ref_img)
    utils.save_features(profile.features_path, ref_kp, ref_desc)

    # ---- debug: reference keypoints + selection montage ---------------
    debug_dir = os.path.join(profile_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    utils.save_image(os.path.join(debug_dir, '00_reference.png'), ref_img)
    utils.save_image(
        os.path.join(debug_dir, '01_reference_keypoints.png'),
        draw_keypoints(ref_gray, ref_kp))

    _save_montage(
        [name_to_path[n] for n in selected],
        os.path.join(debug_dir, '02_teachin_selection.png'),
        highlight=ref_name)

    print(f"\n[init] Profile draft saved: {profile.profile_path}")
    print(f"[init] Debug images: {debug_dir}")
    print(f"\n[init] Next steps:")
    print(f"  1. Define regions (interactively or edit YAML):")
    print(f"     python -m preprocessing_pipeline regions "
          f"--profile {profile_dir}")
    print(f"  2. Or edit regions directly in {profile.profile_path}")

    return profile


# ==================================================================
# Step 2: regions (interactive)
# ==================================================================

def define_regions_interactive(profile_dir: str):
    """Open OpenCV windows to define canonical crop, object mask, etc.

    Regions are written back into the profile YAML.

    Falls back gracefully if no display is available — the operator
    can always set regions by editing profile.yaml.
    """
    from .region_selector import (select_rectangle, select_polygon,
                                  preview_regions, save_region_visualization)

    profile = ProductProfile(profile_dir)
    profile.load()
    ref_img = utils.load_image(profile.reference_image_path)

    print("[regions] Loaded reference image from profile")
    print("[regions] Interactive region selection (OpenCV windows)")
    print()

    regions = profile.data.setdefault('regions', {})

    # ---- canonical crop ----------------------------------------------
    print("=" * 60)
    print("CANONICAL CROP — tight rectangle around the PCB/object.")
    print("  Drag to select, then ENTER/SPACE to confirm.  C to skip.")
    print("=" * 60)
    crop = select_rectangle(ref_img,
                            "CANONICAL CROP — drag, ENTER to confirm")
    if crop:
        regions['canonical_crop'] = crop
        print(f"  -> canonical_crop = {crop}")
    else:
        print("  -> skipped")

    # ---- object mask polygon (optional) ------------------------------
    print()
    print("=" * 60)
    print("OBJECT MASK — polygon around the PCB (optional).")
    print("  Left-click to add points, right-click or ENTER to finish.")
    print("  ESC to skip.")
    print("=" * 60)
    mask = select_polygon(ref_img,
                          "OBJECT MASK — click vertices, right-click to close")
    if mask:
        regions['object_mask'] = mask
        print(f"  -> object_mask: {len(mask)} vertices")
    else:
        print("  -> skipped")

    # ---- similarity ROI (optional) -----------------------------------
    print()
    print("=" * 60)
    print("SIMILARITY ROI — region for quality measurement (optional).")
    print("  If skipped, canonical_crop will be used for quality checks.")
    print("  Drag, ENTER to confirm.  C/ESC to skip.")
    print("=" * 60)
    sim = select_rectangle(ref_img,
                           "SIMILARITY ROI — drag, ENTER to confirm")
    if sim:
        regions['similarity_roi'] = sim
        print(f"  -> similarity_roi = {sim}")
    else:
        print("  -> skipped (will default to canonical_crop)")

    # ---- update profile ----------------------------------------------
    profile.data['regions'] = regions
    if regions.get('canonical_crop'):
        c = regions['canonical_crop']
        profile.data['canonical_size'] = [c[2], c[3]]

    # Re-extract reference features with object mask if defined
    if regions.get('object_mask'):
        print("\n[regions] Re-extracting reference features with object mask…")
        ref_gray = utils.to_gray(ref_img)
        obj_mask = utils.build_mask_from_polygon(
            ref_gray.shape, regions['object_mask'])
        config = profile.get_config()
        aligner = FeatureAligner(config)
        kp, desc = aligner.detect_and_compute(ref_gray, mask=obj_mask)
        utils.save_features(profile.features_path, kp, desc)
        print(f"  -> masked features: {len(kp)}")

    profile.save_data()

    # ---- preview overlay (GUI + file) --------------------------------
    print("\n[regions] Showing region preview …")
    preview_regions(ref_img,
                    canonical_crop=regions.get('canonical_crop'),
                    object_mask=regions.get('object_mask'),
                    similarity_roi=regions.get('similarity_roi'))

    debug_dir = os.path.join(profile_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    save_region_visualization(
        ref_img, regions,
        os.path.join(debug_dir, '03_regions.png'))

    print(f"\n[regions] Regions saved to {profile.profile_path}")
    print(f"[regions] Next: preview alignment quality")
    print(f"  python -m preprocessing_pipeline preview "
          f"--profile {profile_dir}")

    return profile


# ==================================================================
# Step 3: preview
# ==================================================================

def preview_alignment(profile_dir: str,
                      max_images: Optional[int] = None,
                      image_names: Optional[List[str]] = None):
    """Preview alignment on teach-in (or arbitrary) images.

    Aligns each image to the current reference using the profile's
    settings and saves numbered debug visualisations into
    ``<profile_dir>/preview/<stem>/``.

    Returns a list of per-image result dicts.
    """
    profile = ProductProfile(profile_dir)
    profile.load()
    config = profile.get_config()

    ref_img = utils.load_image(profile.reference_image_path)
    ref_gray = utils.to_gray(ref_img)
    ref_kp, ref_desc = utils.load_features(profile.features_path)
    reference_size = tuple(profile.data['reference_size'])

    sim_roi = profile.get_similarity_roi()
    canonical_crop = profile.get_canonical_crop()

    print(f"[preview] reference features: {len(ref_kp)}")
    print(f"[preview] reference size:     {reference_size[0]}x"
          f"{reference_size[1]}")
    if canonical_crop:
        print(f"[preview] canonical_crop:     {canonical_crop}")
    if sim_roi and sim_roi != canonical_crop:
        print(f"[preview] similarity_roi:     {sim_roi}")

    aligner = FeatureAligner(config)

    # ---- resolve image list ------------------------------------------
    dataset_src = profile.data.get('dataset_source', '')
    teachin_names = profile.data.get('teachin_images', [])
    primary_ref = profile.data.get('primary_reference', '')

    names = image_names if image_names else list(teachin_names)
    if max_images and len(names) > max_images:
        names = names[:max_images]

    preview_dir = os.path.join(profile_dir, 'preview')
    os.makedirs(preview_dir, exist_ok=True)
    ref_kp_vis = draw_keypoints(ref_gray, ref_kp)

    print(f"\n[preview] Aligning {len(names)} image(s) …\n")
    results: List[Dict[str, Any]] = []

    for name in names:
        if name == primary_ref:
            print(f"  {name}  — primary reference (skip)")
            continue

        img_path = os.path.join(dataset_src, name)
        if not os.path.exists(img_path):
            print(f"  {name}  — NOT FOUND")
            continue

        print(f"  {name} … ", end="", flush=True)
        image = utils.load_image(img_path)

        result = aligner.align(
            image, ref_gray, ref_kp, ref_desc,
            canonical_size=reference_size,
            similarity_roi=sim_roi,
            canonical_crop=canonical_crop,
            save_debug=True,
        )

        # assemble debug set
        debug = result.debug_images.copy()
        debug['keypoints_ref'] = ref_kp_vis
        warped = debug.get('orientation_selected',
                           debug.get('warp_coarse'))
        if warped is not None:
            debug['warp_overlay'] = draw_warp_overlay(
                ref_gray, utils.to_gray(warped))

        stem = os.path.splitext(name)[0]
        out_dir = os.path.join(preview_dir, stem)
        save_debug_set(debug, out_dir)

        if result.success and result.aligned_image is not None:
            utils.save_image(os.path.join(out_dir, 'aligned.png'),
                             result.aligned_image)
            print(f"OK  sim={result.similarity_score:.3f}  "
                  f"inliers={result.num_inliers}")
        else:
            print(f"FAILED: {result.status}")

        results.append({
            'image': name,
            'status': result.status,
            'similarity': result.similarity_score,
            'num_inliers': result.num_inliers,
            'inlier_ratio': result.inlier_ratio,
        })

    # ---- summary -----------------------------------------------------
    ok = [r for r in results if r['status'] == 'ok']
    fail = [r for r in results if r['status'] != 'ok']
    print(f"\n[preview] {len(ok)} OK / {len(fail)} FAILED "
          f"(out of {len(results)})")
    if ok:
        sims = [r['similarity'] for r in ok]
        inls = [r['num_inliers'] for r in ok]
        print(f"  similarity : min={min(sims):.3f}  "
              f"mean={sum(sims)/len(sims):.3f}  max={max(sims):.3f}")
        print(f"  inliers    : min={min(inls)}  "
              f"mean={sum(inls)/len(inls):.0f}  max={max(inls)}")
    if fail:
        print("  failures:")
        for r in fail:
            print(f"    {r['image']}: {r['status']}")

    print(f"\n[preview] Debug output → {preview_dir}")
    print("[preview] Inspect the results, then:")
    print(f"  • Adjust params in {profile.profile_path} and re-preview, or")
    print(f"  • Finalize: python -m preprocessing_pipeline finalize "
          f"--profile {profile_dir}")

    return results


# ==================================================================
# Step 4: finalize
# ==================================================================

def finalize_profile(profile_dir: str):
    """Build median-consensus reference and lock the profile.

    Aligns every teach-in image to the primary reference, computes
    a pixel-wise median to create a noise-reduced consensus reference,
    re-extracts features, and sets ``finalized: true`` in the profile.
    """
    profile = ProductProfile(profile_dir)
    profile.load()
    config = profile.get_config()

    dataset_src = profile.data.get('dataset_source', '')
    teachin_names = profile.data.get('teachin_images', [])
    primary_ref = profile.data.get('primary_reference', '')
    reference_size = tuple(profile.data['reference_size'])
    sim_roi = profile.get_similarity_roi()
    mask_poly = profile.get_object_mask_polygon()

    print(f"[finalize] profile       = {profile_dir}")
    print(f"[finalize] teach-in imgs = {len(teachin_names)}")
    print(f"[finalize] primary ref   = {primary_ref}")

    ref_img = utils.load_image(profile.reference_image_path)
    ref_gray = utils.to_gray(ref_img)
    ref_kp, ref_desc = utils.load_features(profile.features_path)

    aligner = FeatureAligner(config)

    # ---- align teach-in images (full-frame, no crop) -----------------
    aligned = [ref_img.copy()]
    stats: List[Dict[str, Any]] = []

    for name in teachin_names:
        if name == primary_ref:
            continue
        img_path = os.path.join(dataset_src, name)
        if not os.path.exists(img_path):
            print(f"  {name} — not found, skipped")
            continue

        print(f"  {name} … ", end="", flush=True)
        image = utils.load_image(img_path)

        result = aligner.align(
            image, ref_gray, ref_kp, ref_desc,
            canonical_size=reference_size,
            similarity_roi=sim_roi,
            canonical_crop=None,   # full-frame for median
            save_debug=False,
        )

        if result.success and result.aligned_image is not None:
            aligned.append(result.aligned_image)
            stats.append({
                'image': name,
                'inliers': result.num_inliers,
                'inlier_ratio': result.inlier_ratio,
                'similarity': result.similarity_score,
                'ecc_score': result.ecc_score,
                'flipped': result.orientation_flipped,
            })
            print(f"OK  sim={result.similarity_score:.3f}  "
                  f"inliers={result.num_inliers}")
        else:
            print(f"FAIL ({result.status})")

    # ---- build consensus reference -----------------------------------
    n = len(aligned)
    if n >= 3:
        print(f"\n[finalize] Building median reference from {n} images")
        th, tw = aligned[0].shape[:2]
        stack = []
        for img in aligned:
            if img.shape[:2] == (th, tw):
                stack.append(img)
            else:
                stack.append(cv2.resize(img, (tw, th)))
        consensus = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    elif n == 2:
        print("[finalize] Averaging 2 images for reference")
        consensus = ((aligned[0].astype(np.float64) +
                      aligned[1].astype(np.float64)) / 2).astype(np.uint8)
    else:
        print("[finalize] Using single primary reference (no consensus)")
        consensus = aligned[0]

    # ---- re-extract features on consensus ----------------------------
    consensus_gray = utils.to_gray(consensus)
    feat_mask = None
    if mask_poly:
        feat_mask = utils.build_mask_from_polygon(consensus_gray.shape,
                                                  mask_poly)
    final_kp, final_desc = aligner.detect_and_compute(consensus_gray,
                                                      mask=feat_mask)
    print(f"[finalize] consensus features: {len(final_kp)}")

    # ---- stability warnings -----------------------------------------
    if stats:
        _warn_stability(stats, config)

    # ---- save artefacts ----------------------------------------------
    utils.save_image(profile.reference_image_path, consensus)
    utils.save_features(profile.features_path, final_kp, final_desc)

    canonical_crop = profile.get_canonical_crop()
    if canonical_crop:
        canonical_size = [canonical_crop[2], canonical_crop[3]]
    else:
        canonical_size = list(reference_size)

    profile.data['canonical_size'] = canonical_size
    profile.data['teachin_stats'] = _compute_stats(stats, len(teachin_names))
    profile.data['finalized'] = True
    profile.save_data()

    # ---- debug artefacts ---------------------------------------------
    debug_dir = os.path.join(profile_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    utils.save_image(os.path.join(debug_dir, '04_consensus_reference.png'),
                     consensus)
    utils.save_image(os.path.join(debug_dir, '05_consensus_keypoints.png'),
                     draw_keypoints(consensus_gray, final_kp))
    if canonical_crop:
        x, y, cw, ch = canonical_crop
        utils.save_image(
            os.path.join(debug_dir, '06_canonical_crop_preview.png'),
            consensus[y:y + ch, x:x + cw].copy())

    ts = profile.data['teachin_stats']
    print(f"\n[finalize] DONE — profile finalized")
    print(f"  Profile:        {profile.profile_path}")
    print(f"  Consensus ref:  {profile.reference_image_path}")
    print(f"  Canonical size: {canonical_size[0]}x{canonical_size[1]}")
    if ts.get('mean_similarity'):
        print(f"  Mean similarity: {ts['mean_similarity']:.4f}")
        print(f"  Mean inliers:    {ts['mean_inliers']}")
    print(f"\n  Ready for batch preprocessing:")
    print(f"    python -m preprocessing_pipeline preprocess "
          f"--profile {profile_dir} "
          f"--input <images_dir> --output <output_dir>")

    return profile


# ==================================================================
# Internal helpers
# ==================================================================

def _save_montage(image_paths: List[str], output_path: str,
                  highlight: str = "", cols: int = 5,
                  thumb_size: int = 256):
    """Save a grid montage of images (for reviewing teach-in selection)."""
    thumbs = []
    for p in image_paths:
        img = utils.load_image(p)
        h, w = img.shape[:2]
        s = thumb_size / max(h, w)
        small = cv2.resize(img, (int(w * s), int(h * s)))

        canvas = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
        dy = (thumb_size - small.shape[0]) // 2
        dx = (thumb_size - small.shape[1]) // 2
        canvas[dy:dy + small.shape[0], dx:dx + small.shape[1]] = small

        name = os.path.basename(p)
        colour = (0, 255, 0) if name == highlight else (200, 200, 200)
        label = f"*REF* {name}" if name == highlight else name
        cv2.putText(canvas, label, (4, thumb_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
        if name == highlight:
            cv2.rectangle(canvas, (0, 0),
                          (thumb_size - 1, thumb_size - 1), colour, 3)
        thumbs.append(canvas)

    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))

    montage = np.vstack(rows) if rows else np.zeros((100, 100, 3),
                                                     dtype=np.uint8)
    utils.save_image(output_path, montage)


def _compute_stats(stats: list, total: int) -> Dict[str, Any]:
    if not stats:
        return {'num_teachin_images': total, 'num_aligned': 1}
    inliers = [s['inliers'] for s in stats]
    sims = [s['similarity'] for s in stats]
    return {
        'num_teachin_images': total,
        'num_aligned': len(stats) + 1,
        'mean_inliers': round(sum(inliers) / len(inliers), 1),
        'min_inliers': min(inliers),
        'max_inliers': max(inliers),
        'mean_similarity': round(sum(sims) / len(sims), 4),
        'min_similarity': round(min(sims), 4),
        'max_similarity': round(max(sims), 4),
    }


def _warn_stability(stats: list, config: dict):
    sims = [s['similarity'] for s in stats]
    inliers = [s['inliers'] for s in stats]
    gates = config.get('quality_gates', {})
    min_sim = gates.get('min_similarity_score', 0.3)

    if min(sims) < min_sim:
        print(f"[finalize] WARNING: worst similarity ({min(sims):.3f}) "
              f"< gate ({min_sim:.3f})")
    if max(sims) - min(sims) > 0.3:
        print(f"[finalize] WARNING: large similarity spread "
              f"({min(sims):.3f}–{max(sims):.3f})")
    if min(inliers) < gates.get('min_inlier_count', 10):
        print(f"[finalize] WARNING: min inlier count ({min(inliers)}) "
              f"below threshold")
