"""MANTIS Preprocessing Wizard
==============================
Single-command step-by-step GUI for reviewed teach-in, alignment preview,
parameter tuning, and batch preprocessing.

Launch::

    python -m preprocessing_pipeline wizard

Opens a local web app at http://localhost:8080 with a guided wizard.
"""

import os
import sys
import copy
import base64
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from . import utils
from .config import (ProductProfile, DEFAULT_CONFIG, _CONFIG_KEYS,
                     _deep_copy, _deep_merge)
from .alignment import FeatureAligner
from .debug_viz import (draw_keypoints, draw_matches,
                        draw_orientation_comparison, draw_warp_overlay,
                        save_debug_set)
from .quality import (save_alignment_metadata, generate_summary_csv,
                      print_summary)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ====================================================================
#  Helpers
# ====================================================================

def _enc(img: np.ndarray, max_dim: int = None, q: int = 80) -> str:
    """BGR numpy image -> JPEG data-URI string."""
    if max_dim:
        h, w = img.shape[:2]
        s = min(max_dim / max(h, w), 1.0)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def _enc_full(img: np.ndarray) -> str:
    """BGR numpy image -> PNG data-URI at full resolution."""
    _, buf = cv2.imencode('.png', img)
    return 'data:image/png;base64,' + base64.b64encode(buf).decode()


def _thumb(path: str, size: int = 180) -> np.ndarray:
    """Load image file and return BGR thumbnail."""
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    s = size / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)),
                       interpolation=cv2.INTER_AREA)


def _region_svg(crop, sim, mask, tmp=None, mode=None):
    """Build SVG overlay string showing crop/ROI/mask regions."""
    svg = []
    if crop:
        x, y, w, h = crop
        svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="none" stroke="#00ff00" stroke-width="4"/>')
        svg.append(
            f'<text x="{x+8}" y="{y-10}" fill="#00ff00" '
            f'font-size="28" font-weight="bold">Crop</text>')
    if sim:
        x, y, w, h = sim
        svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="none" stroke="#00ffff" stroke-width="3" '
            f'stroke-dasharray="12,6"/>')
        svg.append(
            f'<text x="{x+8}" y="{y-10}" fill="#00ffff" '
            f'font-size="24">Similarity ROI</text>')
    if mask and len(mask) >= 3:
        pts = ' '.join(f'{p[0]},{p[1]}' for p in mask)
        svg.append(
            f'<polygon points="{pts}" fill="rgba(255,0,255,0.12)" '
            f'stroke="#ff00ff" stroke-width="3"/>')
    # Temporary drawing points
    if tmp:
        clr = {'crop': '#00ff00', 'similarity': '#00ffff',
               'mask': '#ff00ff'}.get(mode, '#ffffff')
        for i, (px, py) in enumerate(tmp):
            svg.append(
                f'<circle cx="{px}" cy="{py}" r="10" '
                f'fill="{clr}" opacity="0.7"/>')
            if mode == 'mask' and i > 0:
                ox, oy = tmp[i - 1]
                svg.append(
                    f'<line x1="{ox}" y1="{oy}" x2="{px}" y2="{py}" '
                    f'stroke="{clr}" stroke-width="2"/>')
    return '\n'.join(svg)


# ====================================================================
#  Wizard state
# ====================================================================

class _State:
    """Mutable bag shared across all wizard steps."""

    def __init__(self):
        self.dataset_path: str = ''
        self.product_id: str = 'pcb1'
        self.image_paths: List[str] = []
        self.name_to_path: Dict[str, str] = {}

        self.selected: List[str] = []
        self.reference_name: str = ''

        self.canonical_crop: Optional[List[int]] = None
        self.similarity_roi: Optional[List[int]] = None
        self.object_mask: Optional[List[List[int]]] = None

        self.config: dict = _deep_copy(DEFAULT_CONFIG)

        self.ref_image: Optional[np.ndarray] = None
        self.ref_gray: Optional[np.ndarray] = None
        self.ref_kp: Optional[list] = None
        self.ref_desc: Optional[np.ndarray] = None

        self.preview_results: List[Dict] = []
        self.profile_dir: str = ''
        self.finalized: bool = False

        self.output_dir: str = ''
        self.batch_records: List[Dict] = []

        # interactive drawing
        self.draw_mode: Optional[str] = None
        self.draw_pts: List[List[int]] = []


# ====================================================================
#  Main wizard
# ====================================================================

def run_wizard(port: int = 8080):
    """Build and launch the NiceGUI wizard application."""
    try:
        from nicegui import ui, events
    except ImportError:
        print('NiceGUI is required for the wizard.')
        print('Install with:  pip install nicegui')
        sys.exit(1)

    S = _State()

    # ── Full-resolution image viewer ─────────────────────────

    def _open_image_viewer(data_uri: str, title: str = 'Image'):
        """Open a full-resolution image in a large dialog with zoom."""
        with ui.dialog() as dlg, ui.card().classes(
                'w-[95vw] h-[92vh] max-w-none p-2'):
            with ui.row().classes(
                    'w-full items-center justify-between mb-1'):
                ui.label(title).classes('text-h6 font-bold')
                with ui.row().classes('gap-1'):
                    zoom_lbl = ui.label('100%').classes(
                        'text-body2 text-grey-4 self-center mr-2')
                    ui.button(icon='zoom_in',
                              on_click=lambda: _zoom(1.25)).props(
                              'flat dense round size=sm')
                    ui.button(icon='zoom_out',
                              on_click=lambda: _zoom(0.8)).props(
                              'flat dense round size=sm')
                    ui.button(icon='fit_screen',
                              on_click=lambda: _zoom_reset()).props(
                              'flat dense round size=sm').tooltip(
                              'Fit to window')
                    ui.button(icon='close', on_click=dlg.close).props(
                        'flat dense round size=sm color=red')
            sa = ui.scroll_area().classes('w-full flex-grow')
            zoom_state = {'level': 1.0}
            with sa:
                viewer_img = ui.image(data_uri).classes(
                    'max-w-none').style(
                    'transform-origin: top left;')

            def _zoom(factor):
                zoom_state['level'] = max(
                    0.1, min(zoom_state['level'] * factor, 10.0))
                pct = int(zoom_state['level'] * 100)
                viewer_img.style(
                    f'transform: scale({zoom_state["level"]}); '
                    f'transform-origin: top left;')
                zoom_lbl.text = f'{pct}%'

            def _zoom_reset():
                zoom_state['level'] = 1.0
                viewer_img.style(
                    'transform: scale(1); '
                    'transform-origin: top left;')
                zoom_lbl.text = '100%'

        dlg.open()

    def _clickable_image(data_uri: str, css_classes: str,
                         title: str = 'Image',
                         full_uri: str = None):
        """Render a clickable thumbnail that opens full-res viewer."""
        show_uri = full_uri if full_uri else data_uri
        img = ui.image(data_uri).classes(
            css_classes + ' cursor-pointer')
        img.on('click', lambda u=show_uri, t=title:
               _open_image_viewer(u, t))
        return img

    # ── helpers that need nicegui import ──────────────────────────

    def _align_one(name: str) -> Dict[str, Any]:
        """Align a single image (runs in thread)."""
        path = S.name_to_path.get(name, '')
        if not os.path.isfile(path):
            return {'name': name, 'status': 'file_not_found',
                    'similarity': 0, 'num_inliers': 0,
                    'inlier_ratio': 0, 'ecc_score': 0, 'flipped': False,
                    'original': None, 'original_full': None,
                    'final': None, 'final_full': None,
                    'debug': {}, 'debug_full': {}}
        image = utils.load_image(path)
        aligner = FeatureAligner(S.config)
        ref_size = (S.ref_gray.shape[1], S.ref_gray.shape[0])
        result = aligner.align(
            image, S.ref_gray, S.ref_kp, S.ref_desc,
            canonical_size=ref_size,
            similarity_roi=S.similarity_roi,
            canonical_crop=S.canonical_crop,
            save_debug=True,
        )
        # encode thumbnails for browser display
        enc_debug = {}
        enc_debug_full = {}
        for k, v in result.debug_images.items():
            enc_debug[k] = _enc(v, max_dim=600)
            enc_debug_full[k] = _enc_full(v)
        orig = _enc(image, max_dim=300)
        orig_full = _enc_full(image)
        final = (_enc(result.aligned_image, max_dim=300)
                 if result.aligned_image is not None else None)
        final_full = (_enc_full(result.aligned_image)
                      if result.aligned_image is not None else None)
        return {
            'name': name,
            'status': result.status,
            'similarity': result.similarity_score,
            'num_inliers': result.num_inliers,
            'inlier_ratio': result.inlier_ratio,
            'ecc_score': result.ecc_score,
            'flipped': result.orientation_flipped,
            'original': orig,
            'original_full': orig_full,
            'final': final,
            'final_full': final_full,
            'debug': enc_debug,
            'debug_full': enc_debug_full,
        }

    # ── page ─────────────────────────────────────────────────────

    @ui.page('/')
    def index():
        ui.dark_mode().enable()
        ui.add_head_html(
            '<style>'
            '.thumb-sel{border:3px solid #4caf50!important}'
            '.thumb-ref{border:3px solid #ffc107!important;'
            'box-shadow:0 0 10px #ffc107}'
            '.thumb-dim{border:2px solid #555!important;opacity:.65}'
            '.cursor-pointer:hover{outline:2px solid #42a5f5;'
            'outline-offset:2px;transition:outline 0.15s}'
            '</style>')

        # header
        with ui.header().classes('items-center gap-3 px-6'):
            ui.icon('precision_manufacturing').classes('text-3xl')
            ui.label('MANTIS Preprocessing Wizard').classes(
                'text-h5 font-bold')
            ui.space()
            ui.label('Step-by-step image alignment for anomaly detection'
                     ).classes('text-subtitle2 text-grey-5')

        # references for cross-step UI elements
        refs: Dict[str, Any] = {}

        with ui.column().classes(
                'w-full max-w-7xl mx-auto px-4 py-2 gap-0'):
            with ui.stepper().props('vertical animated'
                                    ).classes('w-full') as stepper:

                # ═════════════════════════════════════════════════
                #  STEP 1 — Dataset Setup
                # ═════════════════════════════════════════════════
                with ui.step('Dataset Setup'):
                    ui.label('Select the product dataset').classes(
                        'text-h6 mb-1')
                    ui.markdown(
                        'Point to the folder with **normal** images for '
                        'this product.  The wizard will scan this folder '
                        'and select candidate teach-in images '
                        'automatically.')

                    ds_input = ui.input(
                        'Dataset folder path',
                        value='',
                        placeholder='/path/to/dataset/Normal',
                    ).classes('w-full').props('outlined')
                    pid_input = ui.input(
                        'Product ID', value='pcb1',
                    ).classes('w-64').props('outlined')

                    scan_lbl = ui.label('').classes('text-body2 mt-2')
                    scan_thumbs = ui.row().classes('flex-wrap gap-2 mt-1')

                    async def _scan():
                        p = ds_input.value.strip()
                        if not os.path.isdir(p):
                            ui.notify('Folder does not exist',
                                      type='negative')
                            return
                        S.dataset_path = p
                        S.product_id = pid_input.value.strip() or 'pcb1'
                        S.image_paths = utils.list_images(p)
                        if not S.image_paths:
                            ui.notify('No images found', type='negative')
                            scan_lbl.text = 'No images found.'
                            return
                        S.name_to_path = {
                            os.path.basename(x): x for x in S.image_paths}
                        names = sorted(S.name_to_path.keys())
                        n = min(12, len(names))
                        step = max(1, len(names) // n)
                        S.selected = [names[i]
                                      for i in range(0, len(names), step)
                                      ][:n]
                        S.reference_name = S.selected[len(S.selected) // 2]
                        scan_lbl.text = (
                            f'Found {len(S.image_paths)} images.  '
                            f'Auto-selected {len(S.selected)} for teach-in.')
                        scan_thumbs.clear()
                        with scan_thumbs:
                            for nm in S.selected[:6]:
                                ui.image(_enc(_thumb(
                                    S.name_to_path[nm], 120), q=70)
                                ).classes('w-24 h-20 object-contain rounded')
                            if len(S.selected) > 6:
                                ui.label(
                                    f'+{len(S.selected)-6} more'
                                ).classes('self-center text-grey-5')
                        ui.notify(f'{len(S.image_paths)} images found',
                                  type='positive')

                    ui.button('Scan Dataset', on_click=_scan,
                              icon='search').classes('mt-2')

                    # load existing profile
                    with ui.expansion('Load existing profile',
                                      icon='folder_open').classes('mt-4'):
                        prof_input = ui.input(
                            'Profile directory',
                            value=os.path.join(_REPO_ROOT, 'profiles',
                                               'pcb1'),
                        ).classes('w-full').props('outlined dense')

                        async def _load_profile():
                            pd = prof_input.value.strip()
                            pp = ProductProfile(pd)
                            if not pp.exists():
                                ui.notify('Profile not found',
                                          type='negative')
                                return
                            data = pp.load()
                            S.dataset_path = data.get('dataset_source', '')
                            S.product_id = data.get('product_id', 'pcb1')
                            S.selected = data.get('teachin_images', [])
                            S.reference_name = data.get(
                                'primary_reference', '')
                            regions = data.get('regions', {})
                            S.canonical_crop = regions.get('canonical_crop')
                            S.similarity_roi = regions.get('similarity_roi')
                            S.object_mask = regions.get('object_mask')
                            for k in _CONFIG_KEYS:
                                if k in data:
                                    S.config[k] = data[k]
                            S.profile_dir = pd
                            S.finalized = data.get('finalized', False)
                            # scan images
                            if S.dataset_path and os.path.isdir(
                                    S.dataset_path):
                                S.image_paths = utils.list_images(
                                    S.dataset_path)
                                S.name_to_path = {
                                    os.path.basename(x): x
                                    for x in S.image_paths}
                            # load reference
                            if os.path.isfile(pp.reference_image_path):
                                S.ref_image = utils.load_image(
                                    pp.reference_image_path)
                                S.ref_gray = utils.to_gray(S.ref_image)
                            if os.path.isfile(pp.features_path):
                                S.ref_kp, S.ref_desc = utils.load_features(
                                    pp.features_path)
                            ui.notify(
                                f'Profile loaded ({S.product_id}, '
                                f'finalized={S.finalized})',
                                type='positive')

                        ui.button('Load', on_click=_load_profile,
                                  icon='upload').props('flat dense')

                    with ui.stepper_navigation():
                        async def _next1():
                            if not S.image_paths:
                                ui.notify('Scan the dataset first',
                                          type='warning')
                                return
                            stepper.next()
                        ui.button('Next', on_click=_next1,
                                  icon='arrow_forward')

                # ═════════════════════════════════════════════════
                #  STEP 2 — Teach-in Image Selection
                # ═════════════════════════════════════════════════
                with ui.step('Teach-in Images'):
                    ui.label('Choose images for teach-in').classes(
                        'text-h6 mb-1')
                    ui.markdown(
                        'Pick **8–15 sharp, well-exposed** images with '
                        'slight natural variation in brightness and '
                        'position.  Mark one as the **primary reference**.')

                    sel_lbl = ui.label('').classes('text-subtitle2')
                    img_grid = ui.row().classes(
                        'flex-wrap gap-3 mt-2')
                    browse_box = ui.column().classes('w-full mt-4')

                    def _refresh_grid():
                        sel_lbl.text = (
                            f'{len(S.selected)} images selected  ·  '
                            f'Reference: {S.reference_name or "(none)"}')
                        img_grid.clear()
                        with img_grid:
                            for nm in S.selected:
                                p = S.name_to_path.get(nm, '')
                                uri = _enc(_thumb(p, 160), q=75) if p else ''
                                is_ref = nm == S.reference_name
                                cls = 'thumb-ref' if is_ref else 'thumb-sel'
                                with ui.card().classes(
                                        f'p-1 {cls}'):
                                    ui.image(uri).classes(
                                        'w-36 h-28 object-contain')
                                    with ui.row().classes(
                                        'items-center justify-between '
                                        'w-full'):
                                        ui.label(nm).classes('text-xs')
                                        if is_ref:
                                            ui.badge('REF',
                                                     color='amber'
                                                     ).props('dense')
                                    with ui.row().classes('gap-1 mt-1'):
                                        ui.button(
                                            icon='star',
                                            on_click=lambda n=nm: _set_ref(n),
                                        ).props(
                                            'flat dense round size=xs'
                                        ).tooltip('Set as reference')
                                        ui.button(
                                            icon='close',
                                            on_click=lambda n=nm: _rm(n),
                                        ).props(
                                            'flat dense round size=xs '
                                            'color=red'
                                        ).tooltip('Remove')

                    def _set_ref(nm):
                        S.reference_name = nm
                        if nm not in S.selected:
                            S.selected.insert(0, nm)
                        _refresh_grid()

                    def _rm(nm):
                        if nm in S.selected:
                            S.selected.remove(nm)
                        if nm == S.reference_name and S.selected:
                            S.reference_name = S.selected[0]
                        _refresh_grid()

                    def _add(nm):
                        if nm not in S.selected:
                            S.selected.append(nm)
                            _refresh_grid()

                    async def _browse():
                        browse_box.clear()
                        all_names = sorted(S.name_to_path.keys())
                        with browse_box:
                            ui.label(
                                'All dataset images — click to add'
                            ).classes('text-subtitle2')
                            with ui.scroll_area().classes(
                                    'h-72 w-full border rounded'):
                                with ui.row().classes(
                                        'flex-wrap gap-2 p-2'):
                                    for nm in all_names[:80]:
                                        done = nm in S.selected
                                        cls = ('thumb-sel' if done
                                               else 'thumb-dim')
                                        with ui.card().classes(
                                            f'p-1 cursor-pointer {cls}'
                                        ).on('click',
                                             lambda n=nm: _add(n)):
                                            ui.image(_enc(
                                                _thumb(
                                                    S.name_to_path[nm],
                                                    90),
                                                q=60)).classes(
                                                'w-20 h-16 '
                                                'object-contain')
                                            ui.label(nm).classes(
                                                'text-[10px] text-center')
                                    if len(all_names) > 80:
                                        ui.label(
                                            f'… {len(all_names)-80} more'
                                        ).classes('text-grey-5')

                    ui.button('Browse dataset…', on_click=_browse,
                              icon='photo_library').props('flat')

                    with ui.stepper_navigation():
                        ui.button('Back', on_click=stepper.previous
                                  ).props('flat')

                        async def _next2():
                            if not S.selected:
                                ui.notify('Select at least one image',
                                          type='warning')
                                return
                            if not S.reference_name:
                                S.reference_name = S.selected[0]
                            ref_path = S.name_to_path.get(
                                S.reference_name, '')
                            if not os.path.isfile(ref_path):
                                ui.notify('Reference image not found',
                                          type='negative')
                                return
                            S.ref_image = utils.load_image(ref_path)
                            S.ref_gray = utils.to_gray(S.ref_image)
                            aligner = FeatureAligner(S.config)
                            S.ref_kp, S.ref_desc = \
                                aligner.detect_and_compute(S.ref_gray)
                            ui.notify(
                                f'Reference loaded — '
                                f'{len(S.ref_kp)} features',
                                type='positive')
                            _setup_region_image()
                            stepper.next()

                        ui.button('Next', on_click=_next2,
                                  icon='arrow_forward')

                    # auto-refresh on entry
                    stepper.on_value_change(
                        lambda e: (
                            _refresh_grid()
                            if e.value == 'Teach-in Images' else None))

                # ═════════════════════════════════════════════════
                #  STEP 3 — Define Regions
                # ═════════════════════════════════════════════════
                with ui.step('Define Regions'):
                    ui.label(
                        'Define output area and regions'
                    ).classes('text-h6 mb-1')
                    ui.markdown(
                        '**Crop** (green) — The rectangle that becomes '
                        'the final output.  Draw tightly around the '
                        'product.\n\n'
                        '**Similarity ROI** (cyan, optional) — Area '
                        'used to judge alignment quality.  Default = '
                        'same as crop.\n\n'
                        '**Object Mask** (magenta, optional) — Polygon '
                        'around the product.  Helps focus feature '
                        'matching on the product body.')

                    draw_lbl = ui.label(
                        'Click a drawing button, then click on the image.'
                    ).classes('text-subtitle2 text-amber-4')

                    with ui.row().classes('gap-2 flex-wrap'):
                        def _start_crop():
                            S.draw_mode, S.draw_pts = 'crop', []
                            draw_lbl.text = (
                                'CROP: click top-left corner, '
                                'then bottom-right corner')
                            _update_svg()
                            finish_btn.visible = False

                        def _start_sim():
                            S.draw_mode, S.draw_pts = 'similarity', []
                            draw_lbl.text = (
                                'SIMILARITY ROI: click top-left, '
                                'then bottom-right')
                            _update_svg()
                            finish_btn.visible = False

                        def _start_mask():
                            S.draw_mode, S.draw_pts = 'mask', []
                            draw_lbl.text = (
                                'MASK: click vertices around the '
                                'product, then click Finish Mask')
                            finish_btn.visible = True
                            _update_svg()

                        def _finish_mask():
                            if len(S.draw_pts) >= 3:
                                S.object_mask = [
                                    list(p) for p in S.draw_pts]
                                ui.notify(
                                    f'Mask set '
                                    f'({len(S.object_mask)} vertices)',
                                    type='positive')
                            S.draw_mode, S.draw_pts = None, []
                            finish_btn.visible = False
                            draw_lbl.text = 'Done — draw more or Next.'
                            _update_svg()

                        ui.button('Draw Crop', on_click=_start_crop,
                                  icon='crop', color='green')
                        ui.button('Draw Similarity ROI',
                                  on_click=_start_sim,
                                  icon='check_box_outline_blank',
                                  color='cyan')
                        ui.button('Draw Object Mask',
                                  on_click=_start_mask,
                                  icon='pentagon', color='purple')
                        finish_btn = ui.button(
                            'Finish Mask', on_click=_finish_mask,
                            icon='done', color='purple',
                        ).props('outline')
                        finish_btn.visible = False

                    with ui.row().classes('gap-1'):
                        def _rc():
                            S.canonical_crop = None
                            _update_svg()
                            ui.notify('Crop cleared')

                        def _rs():
                            S.similarity_roi = None
                            _update_svg()
                            ui.notify('Similarity ROI cleared')

                        def _rm2():
                            S.object_mask = None
                            S.draw_pts = []
                            _update_svg()
                            ui.notify('Mask cleared')

                        ui.button('Reset Crop', on_click=_rc
                                  ).props('flat dense size=sm')
                        ui.button('Reset ROI', on_click=_rs
                                  ).props('flat dense size=sm')
                        ui.button('Reset Mask', on_click=_rm2
                                  ).props('flat dense size=sm')

                    region_info = ui.label('').classes(
                        'text-body2 text-grey-4')

                    # Container for the interactive image
                    ii_box = ui.column().classes('w-full')
                    refs['ii'] = None

                    def _update_svg():
                        if refs.get('ii'):
                            refs['ii'].content = _region_svg(
                                S.canonical_crop, S.similarity_roi,
                                S.object_mask, S.draw_pts, S.draw_mode)
                        parts = []
                        if S.canonical_crop:
                            x, y, w, h = S.canonical_crop
                            parts.append(f'Crop {w}×{h}')
                        if S.similarity_roi:
                            x, y, w, h = S.similarity_roi
                            parts.append(f'ROI {w}×{h}')
                        if S.object_mask:
                            parts.append(
                                f'Mask {len(S.object_mask)} pts')
                        region_info.text = (
                            '  ·  '.join(parts)
                            if parts else 'No regions yet')

                    def _on_img_click(e):
                        if S.draw_mode is None:
                            return
                        x = int(e.image_x)
                        y = int(e.image_y)
                        S.draw_pts.append([x, y])
                        if S.draw_mode in ('crop', 'similarity'):
                            if len(S.draw_pts) >= 2:
                                x1, y1 = S.draw_pts[0]
                                x2, y2 = S.draw_pts[1]
                                rect = [min(x1, x2), min(y1, y2),
                                        abs(x2 - x1), abs(y2 - y1)]
                                if rect[2] > 10 and rect[3] > 10:
                                    if S.draw_mode == 'crop':
                                        S.canonical_crop = rect
                                    else:
                                        S.similarity_roi = rect
                                    ui.notify(
                                        f'Rectangle set: '
                                        f'{rect[2]}×{rect[3]}',
                                        type='positive')
                                else:
                                    ui.notify(
                                        'Too small — try again',
                                        type='warning')
                                S.draw_mode, S.draw_pts = None, []
                                draw_lbl.text = (
                                    'Done — draw another region '
                                    'or click Next.')
                        _update_svg()

                    def _setup_region_image():
                        ii_box.clear()
                        if S.ref_image is None:
                            return
                        uri = _enc(S.ref_image, max_dim=1400, q=90)
                        with ii_box:
                            ui.button(
                                'View Reference Full Resolution',
                                on_click=lambda: _open_image_viewer(
                                    _enc_full(S.ref_image),
                                    'Reference Image — Full Resolution'),
                                icon='zoom_in',
                            ).props('flat dense size=sm').classes('mb-1')
                            refs['ii'] = ui.interactive_image(
                                uri,
                                on_mouse=_on_img_click,
                                events=['mousedown'],
                                cross=True,
                                content=_region_svg(
                                    S.canonical_crop,
                                    S.similarity_roi,
                                    S.object_mask),
                            ).classes('w-full')

                    with ui.stepper_navigation():
                        ui.button('Back', on_click=stepper.previous
                                  ).props('flat')

                        async def _next3():
                            if S.canonical_crop is None:
                                ui.notify(
                                    'Drawing a crop rectangle is '
                                    'recommended (but not mandatory).',
                                    type='info')
                            # re-extract with mask if defined
                            if S.object_mask and S.ref_gray is not None:
                                mask = utils.build_mask_from_polygon(
                                    S.ref_gray.shape, S.object_mask)
                                aligner = FeatureAligner(S.config)
                                S.ref_kp, S.ref_desc = \
                                    aligner.detect_and_compute(
                                        S.ref_gray, mask=mask)
                                ui.notify(
                                    f'Features re-extracted with mask: '
                                    f'{len(S.ref_kp)}',
                                    type='positive')
                            stepper.next()

                        ui.button('Next', on_click=_next3,
                                  icon='arrow_forward')

                # ═════════════════════════════════════════════════
                #  STEP 4 — Preview & Tune
                # ═════════════════════════════════════════════════
                with ui.step('Preview & Tune'):
                    ui.label(
                        'Preview alignment and adjust parameters'
                    ).classes('text-h6 mb-1')
                    ui.markdown(
                        'Runs the complete alignment pipeline on each '
                        'teach-in image so you can check quality.  '
                        'If results are poor, adjust parameters below '
                        'and re-run.')

                    prev_progress = ui.linear_progress(0).props(
                        'stripe color=cyan')
                    prev_progress.visible = False
                    prev_status = ui.label('').classes('text-body2')
                    prev_box = ui.column().classes('w-full gap-3')

                    async def _run_preview():
                        if S.ref_gray is None:
                            ui.notify('No reference loaded',
                                      type='warning')
                            return
                        todo = [n for n in S.selected
                                if n != S.reference_name
                                and n in S.name_to_path]
                        if not todo:
                            ui.notify('No images to preview',
                                      type='warning')
                            return
                        prev_progress.visible = True
                        prev_progress.value = 0
                        prev_status.text = 'Running preview…'
                        prev_box.clear()
                        S.preview_results = []

                        for i, nm in enumerate(todo):
                            prev_status.text = (
                                f'Aligning {i+1}/{len(todo)}: {nm}')
                            r = await asyncio.to_thread(_align_one, nm)
                            S.preview_results.append(r)
                            prev_progress.value = (i + 1) / len(todo)
                            # build card
                            with prev_box:
                                _build_preview_card(r)

                        prev_progress.visible = False
                        ok = sum(1 for r in S.preview_results
                                 if r['status'] == 'ok')
                        prev_status.text = (
                            f'Done — {ok}/{len(S.preview_results)} OK')

                    ui.button('Run Preview', on_click=_run_preview,
                              icon='play_arrow', color='cyan'
                              ).classes('mb-2')

                    def _build_preview_card(r):
                        ok = r['status'] == 'ok'
                        with ui.card().classes('w-full'):
                            with ui.row().classes(
                                    'items-center gap-4 flex-wrap'):
                                with ui.column().classes('min-w-[140px]'):
                                    ui.label(r['name']).classes(
                                        'text-subtitle1 font-bold')
                                    ui.badge(
                                        r['status'],
                                        color='green' if ok else 'red')
                                    if ok:
                                        ui.label(
                                            f'Similarity: '
                                            f'{r["similarity"]:.3f}'
                                        ).classes('text-body2')
                                        ui.label(
                                            f'Inliers: '
                                            f'{r["num_inliers"]}'
                                        ).classes('text-body2')
                                if r.get('original'):
                                    with ui.column().classes(
                                            'items-center'):
                                        ui.label('Original').classes(
                                            'text-xs text-grey-5')
                                        _clickable_image(
                                            r['original'],
                                            'w-48 h-36 object-contain',
                                            f'{r["name"]} — Original',
                                            r.get('original_full'))
                                if r.get('final'):
                                    with ui.column().classes(
                                            'items-center'):
                                        ui.label('Aligned').classes(
                                            'text-xs text-grey-5')
                                        _clickable_image(
                                            r['final'],
                                            'w-48 h-36 object-contain',
                                            f'{r["name"]} — Aligned',
                                            r.get('final_full'))
                            if r.get('debug'):
                                with ui.expansion(
                                    'Show debug images',
                                    icon='bug_report',
                                ).classes('w-full'):
                                    with ui.row().classes(
                                            'flex-wrap gap-2'):
                                        for k, uri in r['debug'].items():
                                            full = r.get(
                                                'debug_full', {}
                                            ).get(k)
                                            with ui.column().classes(
                                                    'items-center'):
                                                ui.label(k).classes(
                                                    'text-[10px]')
                                                _clickable_image(
                                                    uri,
                                                    'w-44 object-contain',
                                                    f'{r["name"]} — {k}',
                                                    full)

                    # ── Parameter tuning panel ────────────────────
                    ui.separator().classes('my-4')
                    with ui.expansion(
                        'Adjust Parameters',
                        icon='tune',
                    ).classes('w-full'):
                        ui.markdown(
                            'Tweak these if alignment quality is poor.  '
                            'Then click **Re-run Preview** above.')

                        with ui.grid(columns=2).classes('gap-4 w-full'):
                            # Feature detector
                            with ui.column():
                                ui.label('Feature Detector').classes(
                                    'font-bold')
                                ui.label(
                                    'AKAZE is more accurate; ORB is '
                                    'faster.'
                                ).classes('text-xs text-grey-5')
                                det_sel = ui.select(
                                    ['akaze', 'orb'],
                                    value=S.config.get(
                                        'feature_detector', 'akaze'),
                                ).classes('w-48')
                                det_sel.on_value_change(
                                    lambda e: S.config.__setitem__(
                                        'feature_detector', e.value))

                            # Feature sensitivity
                            with ui.column():
                                ui.label(
                                    'Feature Sensitivity'
                                ).classes('font-bold')
                                ui.label(
                                    'Lower = more features detected.  '
                                    'Try lowering if not enough matches.'
                                ).classes('text-xs text-grey-5')
                                sens_sl = ui.slider(
                                    min=0.00005, max=0.002,
                                    step=0.00005,
                                    value=S.config.get(
                                        'feature_params', {}
                                    ).get('akaze', {}).get(
                                        'threshold', 0.0003),
                                ).props('label-always')

                                def _on_sens(e):
                                    S.config.setdefault(
                                        'feature_params', {}
                                    ).setdefault(
                                        'akaze', {}
                                    )['threshold'] = e.value
                                sens_sl.on_value_change(_on_sens)

                            # Matching strictness
                            with ui.column():
                                ui.label(
                                    'Matching Strictness'
                                ).classes('font-bold')
                                ui.label(
                                    'Lower = stricter.  0.70–0.75 works '
                                    'well for most PCBs.'
                                ).classes('text-xs text-grey-5')
                                ratio_sl = ui.slider(
                                    min=0.50, max=0.90, step=0.01,
                                    value=S.config.get(
                                        'matching', {}
                                    ).get('ratio_threshold', 0.75),
                                ).props('label-always')

                                def _on_ratio(e):
                                    S.config.setdefault(
                                        'matching', {}
                                    )['ratio_threshold'] = e.value
                                ratio_sl.on_value_change(_on_ratio)

                            # RANSAC tolerance
                            with ui.column():
                                ui.label(
                                    'RANSAC Tolerance (px)'
                                ).classes('font-bold')
                                ui.label(
                                    'Lower = tighter geometric fit.  '
                                    'Too low may reject valid matches.'
                                ).classes('text-xs text-grey-5')
                                rans_sl = ui.slider(
                                    min=1.0, max=15.0, step=0.5,
                                    value=S.config.get(
                                        'ransac', {}
                                    ).get('reproj_threshold', 5.0),
                                ).props('label-always')

                                def _on_rans(e):
                                    S.config.setdefault(
                                        'ransac', {}
                                    )['reproj_threshold'] = e.value
                                rans_sl.on_value_change(_on_rans)

                            # ECC on/off
                            with ui.column():
                                ui.label('ECC Refinement').classes(
                                    'font-bold')
                                ui.label(
                                    'Sub-pixel refinement.  Usually '
                                    'improves quality.  Turn off only '
                                    'if it causes artifacts.'
                                ).classes('text-xs text-grey-5')
                                ecc_sw = ui.switch(
                                    'Enabled',
                                    value=S.config.get(
                                        'ecc', {}
                                    ).get('enabled', True),
                                )

                                def _on_ecc(e):
                                    S.config.setdefault(
                                        'ecc', {}
                                    )['enabled'] = e.value
                                ecc_sw.on_value_change(_on_ecc)

                            # Quality threshold
                            with ui.column():
                                ui.label(
                                    'Min Quality Score'
                                ).classes('font-bold')
                                ui.label(
                                    'Minimum alignment similarity.  '
                                    'Lower to accept more images, '
                                    'higher to be stricter.'
                                ).classes('text-xs text-grey-5')
                                qgate_sl = ui.slider(
                                    min=0.1, max=0.8, step=0.05,
                                    value=S.config.get(
                                        'quality_gates', {}
                                    ).get('min_similarity_score', 0.3),
                                ).props('label-always')

                                def _on_qgate(e):
                                    S.config.setdefault(
                                        'quality_gates', {}
                                    )['min_similarity_score'] = e.value
                                qgate_sl.on_value_change(_on_qgate)

                    with ui.stepper_navigation():
                        ui.button('Back', on_click=stepper.previous
                                  ).props('flat')
                        ui.button('Next', on_click=stepper.next,
                                  icon='arrow_forward')

                # ═════════════════════════════════════════════════
                #  STEP 5 — Finalize Profile
                # ═════════════════════════════════════════════════
                with ui.step('Finalize'):
                    ui.label('Finalize the alignment profile').classes(
                        'text-h6 mb-1')
                    ui.markdown(
                        'Builds a **median-consensus reference** from '
                        'all aligned teach-in images (reduces sensor '
                        'noise), re-extracts features, and saves '
                        'the profile for production use.')

                    prof_dir_input = ui.input(
                        'Profile save directory',
                        value=os.path.join(_REPO_ROOT, 'profiles',
                                           'pcb1'),
                    ).classes('w-full').props('outlined')

                    fin_progress = ui.linear_progress(0).props(
                        'stripe color=amber')
                    fin_progress.visible = False
                    fin_status = ui.label('').classes('text-body2')
                    fin_done = ui.column()

                    async def _finalize():
                        if S.ref_gray is None:
                            ui.notify('No reference loaded',
                                      type='warning')
                            return
                        pdir = prof_dir_input.value.strip()
                        S.profile_dir = pdir
                        os.makedirs(pdir, exist_ok=True)

                        fin_progress.visible = True
                        fin_progress.value = 0
                        fin_status.text = 'Saving reference…'

                        # save initial reference
                        profile = ProductProfile(pdir)
                        utils.save_image(
                            profile.reference_image_path, S.ref_image)
                        utils.save_features(
                            profile.features_path, S.ref_kp, S.ref_desc)

                        # align teach-in images (full frame) for
                        # consensus
                        ref_size = (S.ref_gray.shape[1],
                                    S.ref_gray.shape[0])
                        aligned_imgs = [S.ref_image.copy()]
                        todo = [n for n in S.selected
                                if n != S.reference_name
                                and n in S.name_to_path]
                        total = len(todo) + 2  # +save, +features

                        for i, nm in enumerate(todo):
                            fin_status.text = (
                                f'Aligning {i+1}/{len(todo)}: {nm}')
                            fin_progress.value = (i + 1) / total

                            def _align_full(nm=nm):
                                img = utils.load_image(
                                    S.name_to_path[nm])
                                a = FeatureAligner(S.config)
                                r = a.align(
                                    img, S.ref_gray,
                                    S.ref_kp, S.ref_desc,
                                    canonical_size=ref_size,
                                    similarity_roi=S.similarity_roi,
                                    canonical_crop=None,
                                    save_debug=False)
                                if r.success and r.aligned_image is not None:
                                    return r.aligned_image
                                return None

                            img = await asyncio.to_thread(_align_full)
                            if img is not None:
                                aligned_imgs.append(img)

                        # build consensus
                        fin_status.text = 'Building consensus reference…'
                        n_aligned = len(aligned_imgs)
                        th, tw = aligned_imgs[0].shape[:2]
                        if n_aligned >= 3:
                            stack = []
                            for im in aligned_imgs:
                                if im.shape[:2] == (th, tw):
                                    stack.append(im)
                                else:
                                    stack.append(
                                        cv2.resize(im, (tw, th)))
                            consensus = np.median(
                                np.stack(stack, axis=0), axis=0
                            ).astype(np.uint8)
                        elif n_aligned == 2:
                            consensus = (
                                (aligned_imgs[0].astype(np.float64)
                                 + aligned_imgs[1].astype(np.float64))
                                / 2
                            ).astype(np.uint8)
                        else:
                            consensus = aligned_imgs[0]

                        # re-extract features
                        fin_status.text = 'Extracting features on consensus…'
                        cons_gray = utils.to_gray(consensus)
                        feat_mask = None
                        if S.object_mask:
                            feat_mask = utils.build_mask_from_polygon(
                                cons_gray.shape, S.object_mask)
                        a2 = FeatureAligner(S.config)
                        final_kp, final_desc = a2.detect_and_compute(
                            cons_gray, mask=feat_mask)

                        S.ref_image = consensus
                        S.ref_gray = cons_gray
                        S.ref_kp = final_kp
                        S.ref_desc = final_desc

                        # save everything
                        fin_status.text = 'Saving profile…'
                        utils.save_image(
                            profile.reference_image_path, consensus)
                        utils.save_features(
                            profile.features_path, final_kp, final_desc)

                        canon_size = list(ref_size)
                        if S.canonical_crop:
                            canon_size = [S.canonical_crop[2],
                                          S.canonical_crop[3]]

                        profile.data = {
                            'product_id': S.product_id,
                            'teachin_mode': 'reviewed',
                            'finalized': True,
                            'created': datetime.now().isoformat(),
                            'dataset_source': os.path.abspath(
                                S.dataset_path),
                            'teachin_images': list(S.selected),
                            'primary_reference': S.reference_name,
                            'reference_image': 'reference.png',
                            'reference_features':
                                'reference_features.npz',
                            'reference_size': list(ref_size),
                            'canonical_size': canon_size,
                            'regions': {
                                'canonical_crop': S.canonical_crop,
                                'object_mask': S.object_mask,
                                'similarity_roi': S.similarity_roi,
                            },
                        }
                        for k in _CONFIG_KEYS:
                            if k in S.config:
                                profile.data[k] = S.config[k]
                        profile.save_data()

                        # debug artefacts
                        dbg = os.path.join(pdir, 'debug')
                        os.makedirs(dbg, exist_ok=True)
                        utils.save_image(
                            os.path.join(dbg, 'consensus_reference.png'),
                            consensus)
                        utils.save_image(
                            os.path.join(dbg, 'consensus_keypoints.png'),
                            draw_keypoints(cons_gray, final_kp))

                        S.finalized = True
                        fin_progress.value = 1.0
                        fin_progress.visible = False
                        fin_status.text = ''

                        fin_done.clear()
                        with fin_done:
                            ui.icon('check_circle', color='green'
                                    ).classes('text-5xl')
                            ui.label(
                                'Profile finalized!'
                            ).classes('text-h6 text-green')
                            ui.label(
                                f'Saved to: {pdir}'
                            ).classes('text-body2')
                            ui.label(
                                f'Consensus from {n_aligned} images  ·  '
                                f'{len(final_kp)} features'
                            ).classes('text-body2 text-grey-5')
                            # Show consensus reference (clickable)
                            ui.label(
                                'Consensus reference — '
                                'click to inspect full resolution'
                            ).classes('text-subtitle2 mt-3')
                            _clickable_image(
                                _enc(consensus, max_dim=500),
                                'w-72 object-contain rounded mt-1',
                                'Consensus Reference',
                                _enc_full(consensus))
                            # Show canonical crop preview if defined
                            if S.canonical_crop:
                                x, y, cw, ch = S.canonical_crop
                                crop_img = consensus[
                                    y:y+ch, x:x+cw].copy()
                                ui.label(
                                    'Canonical crop preview'
                                ).classes('text-subtitle2 mt-2')
                                _clickable_image(
                                    _enc(crop_img, max_dim=500),
                                    'w-72 object-contain rounded mt-1',
                                    'Canonical Crop Preview',
                                    _enc_full(crop_img))

                        ui.notify('Profile finalized!', type='positive')

                    ui.button('Finalize & Save', on_click=_finalize,
                              icon='save', color='amber'
                              ).classes('mt-2')

                    with ui.stepper_navigation():
                        ui.button('Back', on_click=stepper.previous
                                  ).props('flat')
                        ui.button('Next', on_click=stepper.next,
                                  icon='arrow_forward')

                # ═════════════════════════════════════════════════
                #  STEP 6 — Batch Preprocessing
                # ═════════════════════════════════════════════════
                with ui.step('Preprocess'):
                    ui.label(
                        'Run batch preprocessing'
                    ).classes('text-h6 mb-1')
                    ui.markdown(
                        'Process the **entire dataset** using the '
                        'finalized profile.  Each image is aligned, '
                        'cropped, and saved.  Failed images are '
                        'separated automatically.')

                    batch_in = ui.input(
                        'Input image folder',
                        value='',
                        placeholder='/path/to/dataset/Normal',
                    ).classes('w-full').props('outlined')
                    batch_out = ui.input(
                        'Output folder',
                        value=os.path.join(_REPO_ROOT, 'output', 'pcb1'),
                    ).classes('w-full').props('outlined')
                    debug_sw = ui.switch('Save debug images', value=True)

                    batch_progress = ui.linear_progress(0).props(
                        'stripe color=green')
                    batch_progress.visible = False
                    batch_status = ui.label('').classes('text-body2')
                    batch_log = ui.column().classes(
                        'w-full max-h-64 overflow-auto')

                    async def _run_batch():
                        in_dir = batch_in.value.strip()
                        out_dir = batch_out.value.strip()
                        S.output_dir = out_dir

                        if not os.path.isdir(in_dir):
                            ui.notify('Input folder not found',
                                      type='negative')
                            return
                        if S.ref_gray is None:
                            ui.notify('Finalize the profile first',
                                      type='warning')
                            return

                        imgs = utils.list_images(in_dir)
                        if not imgs:
                            ui.notify('No images in input folder',
                                      type='negative')
                            return

                        # prepare dirs
                        aligned_dir = os.path.join(out_dir, 'aligned')
                        meta_dir = os.path.join(out_dir, 'metadata')
                        debug_dir = os.path.join(out_dir, 'debug')
                        failed_dir = os.path.join(out_dir, 'failed')
                        report_dir = os.path.join(out_dir, 'reports')
                        for d in [aligned_dir, meta_dir,
                                  failed_dir, report_dir]:
                            os.makedirs(d, exist_ok=True)
                        do_debug = debug_sw.value
                        if do_debug:
                            os.makedirs(debug_dir, exist_ok=True)

                        batch_progress.visible = True
                        batch_progress.value = 0
                        batch_log.clear()
                        S.batch_records = []
                        total = len(imgs)

                        ref_size = (S.ref_gray.shape[1],
                                    S.ref_gray.shape[0])

                        for i, img_path in enumerate(imgs):
                            name = os.path.basename(img_path)
                            stem = os.path.splitext(name)[0]
                            batch_status.text = (
                                f'Processing {i+1}/{total}: {name}')
                            batch_progress.value = (i + 1) / total

                            def _proc(ip=img_path):
                                try:
                                    im = utils.load_image(ip)
                                except IOError:
                                    return None, 'load_error', {}
                                a = FeatureAligner(S.config)
                                r = a.align(
                                    im, S.ref_gray,
                                    S.ref_kp, S.ref_desc,
                                    canonical_size=ref_size,
                                    similarity_roi=S.similarity_roi,
                                    canonical_crop=S.canonical_crop,
                                    save_debug=do_debug)
                                return r, r.status, r.metadata

                            res, status, meta = \
                                await asyncio.to_thread(_proc)

                            if res is None:
                                rec = {
                                    'image': name, 'status': status,
                                    'num_features': 0, 'num_matches': 0,
                                    'num_inliers': 0,
                                    'inlier_ratio': 0.0,
                                    'similarity': 0.0,
                                    'ecc_score': 0.0,
                                    'orientation_flipped': False}
                            else:
                                rec = {
                                    'image': name,
                                    'status': res.status,
                                    'num_features': meta.get(
                                        'num_features_detected', 0),
                                    'num_matches': meta.get(
                                        'num_matches', 0),
                                    'num_inliers': res.num_inliers,
                                    'inlier_ratio': res.inlier_ratio,
                                    'similarity': res.similarity_score,
                                    'ecc_score': res.ecc_score,
                                    'orientation_flipped':
                                        res.orientation_flipped}

                                if res.success and \
                                        res.aligned_image is not None:
                                    utils.save_image(
                                        os.path.join(aligned_dir,
                                                     f'{stem}.png'),
                                        res.aligned_image)
                                else:
                                    shutil.copy2(
                                        img_path,
                                        os.path.join(failed_dir, name))

                                save_alignment_metadata(
                                    os.path.join(meta_dir,
                                                 f'{stem}.json'),
                                    meta, name, res.status)

                                if do_debug and res.debug_images:
                                    dbg = res.debug_images.copy()
                                    warped = dbg.get(
                                        'orientation_selected',
                                        dbg.get('warp_coarse'))
                                    if warped is not None:
                                        dbg['warp_overlay'] = \
                                            draw_warp_overlay(
                                                S.ref_gray,
                                                utils.to_gray(warped))
                                    save_debug_set(
                                        dbg,
                                        os.path.join(debug_dir, stem))

                            S.batch_records.append(rec)

                            # log line
                            ok = rec['status'] == 'ok'
                            with batch_log:
                                lbl = (
                                    f'{"✅" if ok else "❌"} {name}: '
                                    f'{rec["status"]}')
                                if ok:
                                    lbl += (
                                        f'  sim={rec["similarity"]:.3f}')
                                ui.label(lbl).classes(
                                    'text-xs '
                                    + ('text-green-4'
                                       if ok else 'text-red-4'))

                        # summary CSV
                        csv_path = os.path.join(report_dir,
                                                'summary.csv')
                        generate_summary_csv(S.batch_records, csv_path)

                        batch_progress.visible = False
                        ok_n = sum(1 for r in S.batch_records
                                   if r['status'] == 'ok')
                        batch_status.text = (
                            f'Done — {ok_n}/{total} aligned.  '
                            f'Results in {out_dir}')
                        ui.notify(
                            f'Batch done: {ok_n}/{total} OK',
                            type='positive')

                    ui.button('Start Preprocessing',
                              on_click=_run_batch,
                              icon='play_arrow', color='green'
                              ).classes('mt-2')

                    with ui.stepper_navigation():
                        ui.button('Back', on_click=stepper.previous
                                  ).props('flat')
                        ui.button('Next', on_click=stepper.next,
                                  icon='arrow_forward')

                # ═════════════════════════════════════════════════
                #  STEP 7 — Results
                # ═════════════════════════════════════════════════
                with ui.step('Results'):
                    ui.label('Review results').classes('text-h6 mb-1')
                    ui.markdown(
                        'Check where outputs were saved and inspect '
                        'a sample of aligned images.')

                    res_box = ui.column().classes('w-full gap-3')

                    async def _show_results():
                        res_box.clear()
                        with res_box:
                            if not S.batch_records:
                                ui.label(
                                    'No batch results yet — run '
                                    'preprocessing first.'
                                ).classes('text-grey-5')
                                return

                            total = len(S.batch_records)
                            ok_n = sum(1 for r in S.batch_records
                                       if r['status'] == 'ok')
                            sims = [r['similarity']
                                    for r in S.batch_records
                                    if r['status'] == 'ok']

                            # summary card
                            with ui.card().classes('w-full'):
                                ui.label('Summary').classes(
                                    'text-h6')
                                with ui.grid(columns=3).classes(
                                        'gap-4'):
                                    with ui.column().classes(
                                            'items-center'):
                                        ui.label(str(total)).classes(
                                            'text-3xl font-bold')
                                        ui.label('Total').classes(
                                            'text-xs text-grey-5')
                                    with ui.column().classes(
                                            'items-center'):
                                        ui.label(str(ok_n)).classes(
                                            'text-3xl font-bold '
                                            'text-green')
                                        ui.label('Aligned').classes(
                                            'text-xs text-grey-5')
                                    with ui.column().classes(
                                            'items-center'):
                                        ui.label(
                                            str(total - ok_n)
                                        ).classes(
                                            'text-3xl font-bold '
                                            'text-red')
                                        ui.label('Failed').classes(
                                            'text-xs text-grey-5')
                                if sims:
                                    ui.separator()
                                    ui.label(
                                        f'Similarity — '
                                        f'min: {min(sims):.3f}  '
                                        f'mean: '
                                        f'{sum(sims)/len(sims):.3f}  '
                                        f'max: {max(sims):.3f}'
                                    ).classes('text-body2')

                            # output paths
                            with ui.card().classes('w-full'):
                                ui.label('Output locations').classes(
                                    'text-subtitle1 font-bold')
                                aligned_p = os.path.join(
                                    S.output_dir, 'aligned')
                                ui.label(
                                    f'Aligned images  →  {aligned_p}'
                                ).classes('text-body2')
                                ui.label(
                                    f'Metadata  →  '
                                    f'{os.path.join(S.output_dir, "metadata")}'
                                ).classes('text-body2')
                                ui.label(
                                    f'Reports  →  '
                                    f'{os.path.join(S.output_dir, "reports")}'
                                ).classes('text-body2')
                                if S.profile_dir:
                                    ui.label(
                                        f'Profile  →  {S.profile_dir}'
                                    ).classes('text-body2')

                            # sample gallery
                            aligned_dir = os.path.join(
                                S.output_dir, 'aligned')
                            if os.path.isdir(aligned_dir):
                                samples = sorted(
                                    os.listdir(aligned_dir))[:12]
                                if samples:
                                    ui.label(
                                        'Sample aligned images — '
                                        'click to inspect full resolution'
                                    ).classes(
                                        'text-subtitle1 font-bold mt-2')
                                    ui.label(
                                        'Click any thumbnail to open '
                                        'it full-size with zoom/pan.'
                                    ).classes(
                                        'text-xs text-grey-5 mb-1')
                                    with ui.row().classes(
                                            'flex-wrap gap-2'):
                                        for fn in samples:
                                            fp = os.path.join(
                                                aligned_dir, fn)
                                            t = _thumb(fp, 200)
                                            full_img = cv2.imread(fp)
                                            full_uri = (
                                                _enc_full(full_img)
                                                if full_img is not None
                                                else None)
                                            with ui.column().classes(
                                                    'items-center'):
                                                _clickable_image(
                                                    _enc(t, q=80),
                                                    'w-44 h-36 '
                                                    'object-contain '
                                                    'rounded',
                                                    fn,
                                                    full_uri)
                                                ui.label(fn).classes(
                                                    'text-[10px]')

                    # auto-refresh when step is shown
                    async def _on_results_step(e):
                        if e.value == 'Results':
                            await _show_results()

                    stepper.on_value_change(_on_results_step)

                    ui.button('Refresh results', on_click=_show_results,
                              icon='refresh').classes('mt-2')

                    with ui.stepper_navigation():
                        ui.button('Back', on_click=stepper.previous
                                  ).props('flat')

    ui.run(title='MANTIS Wizard', port=port, reload=False,
           show=True)
