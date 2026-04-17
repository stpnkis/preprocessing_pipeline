"""Quality assessment and reporting for MANTIS preprocessing pipeline."""

import os
import csv
import json
from typing import List, Dict, Any


def save_alignment_metadata(path: str, metadata: Dict[str, Any],
                            image_name: str, status: str):
    """Save per-image alignment metadata as JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = {
        'image': image_name,
        'status': status,
        **metadata,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


_CSV_FIELDS = [
    'image', 'status', 'num_features', 'num_matches', 'num_inliers',
    'inlier_ratio', 'similarity', 'ecc_score', 'orientation_flipped',
]


def generate_summary_csv(records: List[Dict[str, Any]], output_path: str):
    """Generate a CSV summary of all processed images."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if not records:
        return
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def print_summary(records: List[Dict[str, Any]]):
    """Print a human-readable summary of batch processing results."""
    total = len(records)
    ok = sum(1 for r in records if r.get('status') == 'ok')
    failed = total - ok

    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total images:    {total}")
    print(f"Successful:      {ok}")
    print(f"Failed:          {failed}")

    if ok > 0:
        sims = [r['similarity'] for r in records if r['status'] == 'ok']
        inls = [r['num_inliers'] for r in records if r['status'] == 'ok']
        print(f"\nAlignment quality (successful images):")
        print(f"  Similarity:  min={min(sims):.3f}  "
              f"mean={sum(sims)/len(sims):.3f}  max={max(sims):.3f}")
        print(f"  Inliers:     min={min(inls)}  "
              f"mean={sum(inls)/len(inls):.0f}  max={max(inls)}")

    if failed > 0:
        print(f"\nFailure reasons:")
        reasons: Dict[str, int] = {}
        for r in records:
            if r.get('status') != 'ok':
                reasons[r['status']] = reasons.get(r['status'], 0) + 1
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")

    print(f"{'=' * 60}\n")
