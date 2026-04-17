"""Configuration and profile management for MANTIS preprocessing pipeline."""

import os
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List


DEFAULT_CONFIG: Dict[str, Any] = {
    'product_id': 'pcb1',

    'feature_detector': 'akaze',
    'feature_params': {
        'akaze': {
            'descriptor_type': 5,    # cv2.AKAZE_DESCRIPTOR_MLDB
            'threshold': 0.0003,     # lower → more features
        },
        'orb': {
            'nfeatures': 5000,
            'scale_factor': 1.2,
            'nlevels': 8,
        },
    },

    'matching': {
        'ratio_threshold': 0.75,     # Lowe's ratio test
        'min_matches': 15,           # minimum good matches required
    },

    'ransac': {
        'reproj_threshold': 5.0,     # pixels
        'max_iters': 5000,
        'confidence': 0.999,
    },

    'ecc': {
        'enabled': True,
        'motion_type': 'euclidean',  # euclidean | affine | translation | homography
        'max_iterations': 200,
        'epsilon': 1e-5,
        'gaussian_filter_size': 5,
    },

    'orientation': {
        'check_180': True,
    },

    'quality_gates': {
        'min_inlier_count': 10,
        'min_inlier_ratio': 0.15,
        'min_similarity_score': 0.3,
        'max_warp_area_ratio': 10.0,   # reject if warped area / original area > this
        'min_warp_area_ratio': 0.1,    # reject if warped area / original area < this
    },

    'output': {
        'save_debug': True,
        'image_format': 'png',
    },
}

# Keys in profile YAML that store alignment configuration
_CONFIG_KEYS = [
    'feature_detector', 'feature_params', 'matching',
    'ransac', 'ecc', 'orientation', 'quality_gates', 'output',
]


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file, merging with defaults."""
    config = _deep_copy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    return config


def _deep_copy(d: dict) -> dict:
    """Simple recursive deep copy for nested dicts."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = _deep_copy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ProductProfile:
    """Manages a product alignment profile (teach-in output)."""

    def __init__(self, profile_dir: str):
        self.profile_dir = profile_dir
        self.profile_path = os.path.join(profile_dir, 'profile.yaml')
        self.reference_image_path = os.path.join(profile_dir, 'reference.png')
        self.features_path = os.path.join(profile_dir, 'reference_features.npz')
        self.data: Dict[str, Any] = {}

    def exists(self) -> bool:
        """Check if profile YAML exists on disk."""
        return os.path.exists(self.profile_path)

    def save(self, config: Dict[str, Any], reference_size: Tuple[int, int],
             canonical_size: Tuple[int, int], roi: Optional[list],
             teachin_stats: Dict[str, Any]):
        """Save profile to disk.

        reference_size and canonical_size are (width, height).
        """
        os.makedirs(self.profile_dir, exist_ok=True)

        self.data = {
            'product_id': config.get('product_id', 'unknown'),
            'created': datetime.now().isoformat(),
            'reference_image': 'reference.png',
            'reference_features': 'reference_features.npz',
            'reference_size': list(reference_size),
            'canonical_size': list(canonical_size),
            'roi': roi,

            'feature_detector': config['feature_detector'],
            'feature_params': config['feature_params'],
            'matching': config['matching'],
            'ransac': config['ransac'],
            'ecc': config['ecc'],
            'orientation': config['orientation'],
            'quality_gates': config['quality_gates'],
            'output': config['output'],

            'teachin_stats': teachin_stats,
        }

        with open(self.profile_path, 'w') as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

    def load(self) -> Dict[str, Any]:
        """Load profile from disk."""
        with open(self.profile_path, 'r') as f:
            self.data = yaml.safe_load(f) or {}
        return self.data

    def save_data(self):
        """Save current self.data directly to YAML (for reviewed workflow)."""
        os.makedirs(self.profile_dir, exist_ok=True)
        self.data['updated'] = datetime.now().isoformat()
        with open(self.profile_path, 'w') as f:
            yaml.dump(self.data, f, default_flow_style=False,
                      sort_keys=False)

    def get_config(self) -> Dict[str, Any]:
        """Extract alignment config from loaded profile, merged with defaults."""
        config = _deep_copy(DEFAULT_CONFIG)
        for key in _CONFIG_KEYS:
            if key in self.data:
                if (isinstance(self.data[key], dict)
                        and isinstance(config.get(key), dict)):
                    config[key] = _deep_merge(config[key], self.data[key])
                else:
                    config[key] = self.data[key]
        config['product_id'] = self.data.get('product_id', 'unknown')
        return config

    def get_regions(self) -> Dict[str, Any]:
        """Get region definitions from profile."""
        return self.data.get('regions', {})

    def get_similarity_roi(self) -> Optional[list]:
        """Get effective similarity ROI (for orientation, ECC, quality).

        Falls back: similarity_roi -> canonical_crop -> legacy roi.
        """
        regions = self.get_regions()
        roi = regions.get('similarity_roi')
        if roi is None:
            roi = regions.get('canonical_crop')
        if roi is None:
            roi = self.data.get('roi')  # backward compat
        return roi

    def get_canonical_crop(self) -> Optional[list]:
        """Get canonical crop region for final output.

        Falls back: canonical_crop -> legacy roi.
        """
        regions = self.get_regions()
        crop = regions.get('canonical_crop')
        if crop is None:
            crop = self.data.get('roi')  # backward compat
        return crop

    def get_object_mask_polygon(self) -> Optional[list]:
        """Get object mask polygon vertices, or None."""
        return self.get_regions().get('object_mask')
