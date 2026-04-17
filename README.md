# MANTIS Preprocessing Pipeline

Production-grade image alignment and preprocessing for industrial anomaly
detection (PatchCore, RD++, AnomalyDINO, etc.).

Given a set of normal product images, the pipeline:
1. builds a **reviewed, reusable product profile** via a controlled teach-in,
2. **aligns every image** to a canonical reference using feature-based
   registration (AKAZE/ORB + RANSAC homography + optional ECC refinement),
3. **crops** the aligned result to a tight canonical frame around the product,
4. produces **metadata, debug visualisations, and summary reports**.

The goal is to minimise pose-induced variance (rotation, translation, scale,
placement jitter) so that downstream anomaly detectors focus on real defects
instead of geometric noise.

---

## Installation

```bash
git clone https://github.com/stpnkis/preprocessing_pipeline.git
cd preprocessing_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `opencv-python >= 4.5`, `numpy >= 1.20`, `pyyaml >= 5.0`,
`nicegui >= 1.4` (for the GUI wizard).

---

## Quick Start — GUI Wizard (Recommended)

The easiest way to use MANTIS is the **one-command GUI wizard**.
It guides you through every step visually — no YAML editing, no
memorising commands.

```bash
python -m preprocessing_pipeline wizard
```

This opens a local web app in your browser at `http://localhost:8080`.

---

## Operator Guide — Wizard Step by Step

### Step 1: Dataset Setup

**What you see:** A text field for the dataset folder path and a product ID.

**What to do:**
1. Enter (or confirm) the path to the folder containing **normal** images.
   For example:
   `/path/to/your/dataset/Normal`
2. Enter a product ID (e.g. `pcb1`).
3. Click **Scan Dataset**.
4. The wizard reports how many images were found and auto-selects ~12
   evenly spaced images for teach-in.
5. Click **Next**.

> **Tip:** If you already have a saved profile from a previous run, expand
> "Load existing profile", enter the profile directory, and click Load.

### Step 2: Teach-in Images

**What it means:** These images define what "normal" looks like.  The
alignment algorithm learns from them.

**What to do:**
1. Review the auto-selected images shown as thumbnails.
2. **Remove** bad images (blurry, poorly exposed, occluded) by clicking
   the red ✕ button.
3. **Add** more images by clicking "Browse dataset…" and clicking on
   any image in the grid.
4. **Set the reference:** Click the ★ button on the sharpest, most
   centered, best-exposed image.  This image becomes the alignment
   target — every other image will be warped to match it.
5. Aim for **8–15 images**.  More = better noise reduction later.
6. Click **Next**.

> **How to choose good teach-in images:**
> - Sharp, well-lit, product clearly visible.
> - Include slight natural variation (brightness, small position shifts).
> - Avoid extremes (very dark, very bright, rotated 90°, partially
>   occluded).

### Step 3: Define Regions

**What it means:** You tell the system which part of the image matters.

**What to do:**

You draw regions on the reference image by clicking directly on it.
There are three region types:

| Region | Colour | What it does |
|--------|--------|--------------|
| **Crop** | Green | The final output rectangle.  Everything outside is discarded. |
| **Similarity ROI** | Cyan (dashed) | Area used to judge alignment quality.  Optional — defaults to crop. |
| **Object Mask** | Magenta | Polygon around the product body.  Focuses feature matching.  Optional. |

**Drawing a rectangle (Crop or Similarity ROI):**
1. Click the **Draw Crop** (or **Draw Similarity ROI**) button.
2. Click the **top-left corner** of the rectangle on the image.
3. Click the **bottom-right corner**.
4. The rectangle appears immediately.

**Drawing a polygon (Object Mask):**
1. Click **Draw Object Mask**.
2. Click each vertex along the product outline.
3. Click **Finish Mask** when done (minimum 3 points).

**Reset:** Use the Reset buttons to clear a region and start over.

> **Best practice for PCB1:**
> - **Crop tightly** around the PCB.  Background = false anomalies.
> - Include the full product body plus a thin margin (20–40 px).
> - For Similarity ROI, pick the most feature-rich area (e.g. chip
>   labels, connector pins) — not a uniform flat surface.

Click **Next** when done.

### Step 4: Preview & Tune

**What it means:** The system aligns each teach-in image using your
current settings so you can see whether the result is good.

**What to do:**
1. Click **Run Preview**.
2. Wait while each image is aligned (progress bar shown).
3. Review the result cards:
   - **✅ OK** images show the original and aligned side by side.
   - **❌ FAILED** images show what went wrong.
   - Expand "Show debug images" for detailed pipeline steps.
4. Look at the **similarity scores** — above 0.6 is good, above 0.8
   is excellent.

**If results are poor:**
1. Expand **Adjust Parameters** at the bottom.
2. Change settings using plain-language controls:

| Control | What it changes |
|---------|-----------------|
| Feature Detector | AKAZE (more accurate) vs ORB (faster) |
| Feature Sensitivity | Lower → more features detected |
| Matching Strictness | Lower → stricter matching (fewer but better) |
| RANSAC Tolerance | Lower → tighter geometric fit |
| ECC Refinement | On/Off — sub-pixel alignment improvement |
| Min Quality Score | Lower to accept more images |

3. Click **Run Preview** again to see the effect.
4. Iterate until you're satisfied.

Click **Next** when preview quality is good.

### Step 5: Finalize

**What it means:** The system builds a **consensus reference** by
averaging all aligned teach-in images (reduces sensor noise), then
saves the complete profile for production use.

**What to do:**
1. Confirm the profile save directory (default: `profiles/pcb1/`).
2. Click **Finalize & Save**.
3. Wait for the progress to complete.
4. You'll see a green confirmation with the number of features
   extracted on the consensus reference.

The profile is now ready for batch preprocessing.

### Step 6: Preprocess

**What it means:** The system processes the **entire dataset** using
the finalized profile.  Each image is aligned, cropped, and saved.

**What to do:**
1. Confirm input folder (the dataset path) and output folder
   (default: `output/pcb1/`).
2. Toggle "Save debug images" on or off (off = faster, less disk).
3. Click **Start Preprocessing**.
4. Watch the progress bar and per-image log.
5. When done, you see a summary (e.g. "380/400 aligned").

### Step 7: Results

**What it means:** Review where everything was saved and inspect
sample outputs.

**What you see:**
- **Summary:** total / aligned / failed counts, similarity statistics.
- **Output locations:** exact folder paths for aligned images,
  metadata, reports, and the profile.
- **Sample gallery:** thumbnails of the first 12 aligned images.

**After the wizard:**
- Browse `output/pcb1/aligned/` to see all canonical images.
- Open `output/pcb1/reports/summary.csv` for per-image metrics.
- Failed images are in `output/pcb1/failed/`.
- Debug visualisations are in `output/pcb1/debug/` (per image).

---

## Quick Start (Reviewed Teach-in Workflow)

The CLI workflow is an alternative to the wizard for advanced users.
All commands are run from the repository root (`preprocessing_pipeline/`).

### Step 1 — Initialise the profile

```bash
python -m preprocessing_pipeline init \
    --dataset /path/to/your/dataset/Normal \
    --profile profiles/pcb1 \
    --product-id pcb1 \
    --num-images 12
```

This scans the dataset, auto-selects 12 evenly spaced teach-in images,
picks a primary reference (middle image), extracts features on it, and
writes a **draft profile** to `profiles/pcb1/`.

**Options:**
| Flag | Description |
|------|-------------|
| `--num-images N` | Auto-select N evenly spaced images (default 10) |
| `--images A.JPG B.JPG …` | Manually specify teach-in images |
| `--reference X.JPG` | Choose a specific primary reference |
| `--config custom.yaml` | Override default alignment parameters |

**Output:**
```
profiles/pcb1/
  profile.yaml              ← editable profile (all settings)
  reference.png             ← primary reference image
  reference_features.npz    ← keypoints + descriptors
  debug/
    00_reference.png
    01_reference_keypoints.png
    02_teachin_selection.png  ← montage of selected images
```

### Step 2 — Define regions

You have two options:

**Option A: Interactive (requires display)**
```bash
python -m preprocessing_pipeline regions --profile profiles/pcb1
```
Opens OpenCV windows for selecting:
1. **Canonical crop** — tight rectangle around the product (green)
2. **Object mask** — polygon around the product body (optional, magenta)
3. **Similarity ROI** — quality measurement region (optional, cyan)

**Option B: Edit YAML directly**
Open `profiles/pcb1/profile.yaml` and set the `regions:` section:
```yaml
regions:
  canonical_crop: [180, 210, 1030, 550]   # [x, y, width, height]
  object_mask: null                        # or [[x1,y1], [x2,y2], …]
  similarity_roi: [220, 310, 940, 400]    # null → defaults to canonical_crop
```

**Region meanings:**
| Region | Purpose | Effect |
|--------|---------|--------|
| `canonical_crop` | Final output rectangle | Aligned images are cropped to this box |
| `object_mask` | Polygon around product body | Reference features extracted only inside mask |
| `similarity_roi` | Quality measurement area | Used for orientation check, ECC, quality gates |

### Step 3 — Preview alignment

```bash
python -m preprocessing_pipeline preview --profile profiles/pcb1
```

Aligns each teach-in image and saves numbered debug visualisations to
`profiles/pcb1/preview/<image>/`:

```
01_keypoints_ref.png          ← reference keypoints
02_keypoints_cur.png          ← current image keypoints
03_matches.png                ← inlier feature matches
04_warp_coarse.png            ← homography warp result
05_orientation_comparison.png ← 0° vs 180° candidates with NCC scores
06_orientation_selected.png   ← chosen orientation
07_warp_overlay.png           ← blended reference + warped
08_ecc_refined.png            ← after ECC sub-pixel refinement
09_final.png                  ← final canonical crop
aligned.png                   ← the actual output image
```

Inspect these. If quality is poor, edit `profile.yaml` parameters and
re-run preview. Iterate until satisfied.

**Options:**
| Flag | Description |
|------|-------------|
| `--max-images N` | Preview only N images |
| `--images A.JPG B.JPG` | Preview specific images |

### Step 4 — Finalize

```bash
python -m preprocessing_pipeline finalize --profile profiles/pcb1
```

Aligns all teach-in images to the primary reference, builds a
**median-consensus reference** (reduces sensor noise), re-extracts features,
and sets `finalized: true` in the profile.

After this step the profile is production-ready.

---

## Batch Preprocessing

```bash
python -m preprocessing_pipeline preprocess \
    --profile profiles/pcb1 \
    --input  /path/to/your/dataset/Normal \
    --output output/pcb1
```

Add `--no-debug` to skip per-image debug images (significantly faster).

**Output structure:**
```
output/pcb1/
  aligned/       ← canonical aligned images (PNG), ready for training
  metadata/      ← per-image JSON with alignment metrics
  debug/         ← per-image debug visualisations (if enabled)
  failed/        ← copies of images that failed alignment
  reports/
    summary.csv  ← one row per image with all metrics
```

**Summary CSV columns:**
`image, status, num_features, num_matches, num_inliers, inlier_ratio,
similarity, ecc_score, orientation_flipped`

---

## Legacy Automatic Teach-in

The original single-command teach-in is preserved:

```bash
python -m preprocessing_pipeline teachin \
    --input  /path/to/teachin_images \
    --output profiles/pcb1 \
    --roi 100 50 800 600
```

This is a quick-start option that runs all steps automatically without
review. The reviewed workflow above is recommended for production use.

---

## Alignment Pipeline Stages

```
Input image
  │
  ├─ 1. Feature detection (AKAZE / ORB)
  ├─ 2. Descriptor matching (KNN + Lowe's ratio test)
  ├─ 3. Homography estimation (RANSAC)
  │      └─ quality gates: inlier count, ratio, geometric sanity
  ├─ 4. Perspective warp to reference frame
  ├─ 5. 180° orientation resolution (NCC comparison)
  ├─ 6. ECC sub-pixel refinement (optional, Euclidean model)
  └─ 7. Canonical crop → output
```

---

## Tunable Parameters

All parameters live in `profile.yaml` and can be edited between preview runs.

### Feature Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_detector` | `akaze` | `akaze` or `orb` |
| `feature_params.akaze.threshold` | `0.0003` | Lower → more features |
| `feature_params.orb.nfeatures` | `5000` | Max ORB features |

### Matching
| Parameter | Default | Description |
|-----------|---------|-------------|
| `matching.ratio_threshold` | `0.75` | Lowe's ratio test (lower → stricter) |
| `matching.min_matches` | `15` | Minimum good matches to proceed |

### RANSAC
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ransac.reproj_threshold` | `5.0` | Reprojection tolerance (px) |
| `ransac.max_iters` | `5000` | Maximum RANSAC iterations |
| `ransac.confidence` | `0.999` | RANSAC confidence level |

### ECC Refinement
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ecc.enabled` | `true` | Enable sub-pixel refinement |
| `ecc.motion_type` | `euclidean` | `euclidean` / `affine` / `translation` |
| `ecc.max_iterations` | `200` | ECC iteration limit |

### Orientation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `orientation.check_180` | `true` | Resolve 180° ambiguity via NCC |

### Quality Gates
| Parameter | Default | Description |
|-----------|---------|-------------|
| `quality_gates.min_inlier_count` | `10` | Min RANSAC inliers |
| `quality_gates.min_inlier_ratio` | `0.15` | Min inlier/match ratio |
| `quality_gates.min_similarity_score` | `0.3` | Min NCC after warp |
| `quality_gates.max_warp_area_ratio` | `10.0` | Geometric sanity bound |
| `quality_gates.min_warp_area_ratio` | `0.1` | Geometric sanity bound |

---

## Understanding Success / Failure

### Successful alignment
- `status: ok` in metadata
- `similarity` > 0.6 is good; > 0.8 is excellent
- `inlier_ratio` > 0.3 indicates strong feature agreement
- Output is in `aligned/`

### Failure statuses
| Status | Meaning | Typical fix |
|--------|---------|-------------|
| `insufficient_matches` | Too few feature matches | Lower ratio_threshold, increase features |
| `homography_failed` | RANSAC could not find homography | Check image quality, more features |
| `low_inlier_count` | Not enough RANSAC inliers | Lower min_inlier_count or improve matching |
| `low_inlier_ratio` | Too many outliers | Stricter ratio test, check for occlusion |
| `unreasonable_homography` | Geometric sanity failed | Image may be very different from reference |
| `low_similarity` | Warped image doesn't match reference | Check orientation, ROI, image quality |

---

## Best Practices for PCB1

### How many teach-in images?
- **8–15 images** is a good range.
- Cover the natural variation in brightness, slight position shifts, etc.
- More images → better median consensus → more noise reduction.

### Choosing a good reference
- Pick a sharp, well-exposed, centered image.
- Avoid images at the extremes of position/brightness variation.
- The middle-of-dataset auto-selection usually works well.

### Canonical crop for PatchCore
- Crop tightly around the product. **Background = false anomaly source.**
- Include the full product body plus a thin margin (20–40 px).
- Include functional features (pins, connectors) that anomaly detectors
  should monitor.
- Do NOT include large background areas — PatchCore treats any
  inconsistent background as an anomaly.

### Similarity ROI
- Set this to the most feature-rich, most distinctive region of the product.
- Exclude areas that vary naturally (e.g., reflective surfaces, moving parts).
- A well-chosen similarity ROI improves orientation detection and quality gating.

### Detecting poor teach-in settings
- Preview similarity < 0.5 → crop, reference, or parameters need adjustment.
- Large similarity spread (max−min > 0.2) → some teach-in images are
  problematic; consider removing them.
- Very few features (< 200) → lower the detector threshold or switch
  detector type.
- Many 180° flips → the similarity ROI may not be distinctive enough;
  choose an asymmetric region.

### Improving alignment quality
1. Lower `feature_params.akaze.threshold` to get more features.
2. Tighten `matching.ratio_threshold` (e.g., 0.70) for fewer but
   better matches.
3. Lower `ransac.reproj_threshold` (e.g., 3.0) for tighter geometric fit.
4. Use an `object_mask` polygon to exclude background from reference
   features — reduces false matches.
5. If ECC degrades quality (rare), set `ecc.enabled: false`.

---

## Module Structure

| Module | Responsibility |
|--------|---------------|
| `wizard.py` | **GUI wizard** — NiceGUI step-by-step teach-in + preprocessing |
| `pipeline.py` | CLI entry point, command dispatch |
| `reviewed_teachin.py` | 4-step reviewed teach-in workflow (CLI backend) |
| `region_selector.py` | Interactive OpenCV region selection tools (CLI fallback) |
| `alignment.py` | Feature-based alignment engine |
| `config.py` | Profile I/O, configuration management |
| `teachin.py` | Legacy automatic teach-in |
| `preprocess.py` | Batch preprocessing with saved profile |
| `quality.py` | Metadata, CSV reports, summaries |
| `debug_viz.py` | Debug visualisation helpers |
| `utils.py` | Image I/O, feature serialisation |
