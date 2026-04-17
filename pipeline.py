"""MANTIS Preprocessing Pipeline — CLI entry point.

Reviewed teach-in workflow (recommended):

    python -m preprocessing_pipeline init      --dataset ... --profile ...
    python -m preprocessing_pipeline regions   --profile ...
    python -m preprocessing_pipeline preview   --profile ...
    python -m preprocessing_pipeline finalize  --profile ...

Batch processing:

    python -m preprocessing_pipeline preprocess --profile ... --input ... --output ...

Legacy automatic teach-in:

    python -m preprocessing_pipeline teachin   --input ... --output ...
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description=(
            'MANTIS Preprocessing Pipeline for Industrial Anomaly Detection'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )

    sub = parser.add_subparsers(dest='command', help='Pipeline command')

    # ---- init (reviewed teach-in step 1) -----------------------------
    p_init = sub.add_parser(
        'init',
        help='Initialise a reviewed teach-in profile')
    p_init.add_argument(
        '--dataset', '-d', required=True,
        help='Directory with normal images (full dataset)')
    p_init.add_argument(
        '--profile', '-p', required=True,
        help='Profile output directory')
    p_init.add_argument(
        '--config', '-c', default=None,
        help='Optional YAML config override')
    p_init.add_argument(
        '--product-id', default='pcb1',
        help='Product identifier (default: pcb1)')
    p_init.add_argument(
        '--num-images', type=int, default=10,
        help='Auto-select this many evenly spaced teach-in images '
             '(default: 10)')
    p_init.add_argument(
        '--images', nargs='+', default=None,
        metavar='FILE',
        help='Explicit teach-in image filenames (overrides --num-images)')
    p_init.add_argument(
        '--reference', default=None,
        help='Filename of the primary reference image')

    # ---- regions (reviewed teach-in step 2) --------------------------
    p_reg = sub.add_parser(
        'regions',
        help='Define ROI / mask / canonical crop interactively')
    p_reg.add_argument(
        '--profile', '-p', required=True,
        help='Profile directory')

    # ---- preview (reviewed teach-in step 3) --------------------------
    p_pv = sub.add_parser(
        'preview',
        help='Preview alignment quality on teach-in images')
    p_pv.add_argument(
        '--profile', '-p', required=True,
        help='Profile directory')
    p_pv.add_argument(
        '--max-images', type=int, default=None,
        help='Limit number of images to preview')
    p_pv.add_argument(
        '--images', nargs='+', default=None,
        metavar='FILE',
        help='Specific image filenames to preview')

    # ---- finalize (reviewed teach-in step 4) -------------------------
    p_fin = sub.add_parser(
        'finalize',
        help='Build consensus reference and lock the profile')
    p_fin.add_argument(
        '--profile', '-p', required=True,
        help='Profile directory')

    # ---- teachin (legacy automatic) ----------------------------------
    p_ti = sub.add_parser(
        'teachin',
        help='Automatic teach-in (legacy mode)')
    p_ti.add_argument('--input', '-i', required=True,
                      help='Directory with normal teach-in images')
    p_ti.add_argument('--output', '-o', required=True,
                      help='Directory to save product profile')
    p_ti.add_argument('--config', '-c', default=None,
                      help='Optional YAML config override file')
    p_ti.add_argument('--product-id', default='pcb1',
                      help='Product identifier (default: pcb1)')
    p_ti.add_argument('--roi', nargs=4, type=int,
                      metavar=('X', 'Y', 'W', 'H'),
                      help='Region of interest [x y width height]')

    # ---- wizard (GUI) ------------------------------------------------
    p_wiz = sub.add_parser(
        'wizard',
        help='Launch the step-by-step GUI wizard (recommended)')
    p_wiz.add_argument(
        '--port', type=int, default=8080,
        help='Local web server port (default: 8080)')

    # ---- preprocess --------------------------------------------------
    p_pp = sub.add_parser(
        'preprocess',
        help='Batch-preprocess images using a finalized profile')
    p_pp.add_argument('--profile', '-p', required=True,
                      help='Profile directory')
    p_pp.add_argument('--input', '-i', required=True,
                      help='Directory with images to process')
    p_pp.add_argument('--output', '-o', required=True,
                      help='Root output directory')
    p_pp.add_argument('--no-debug', action='store_true',
                      help='Disable debug visualisation output')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    _DISPATCH = {
        'wizard':     _cmd_wizard,
        'init':       _cmd_init,
        'regions':    _cmd_regions,
        'preview':    _cmd_preview,
        'finalize':   _cmd_finalize,
        'teachin':    _cmd_teachin,
        'preprocess': _cmd_preprocess,
    }
    _DISPATCH[args.command](args)


# ------------------------------------------------------------------
# Command handlers (lazy imports)
# ------------------------------------------------------------------

def _cmd_wizard(args):
    from .wizard import run_wizard
    run_wizard(port=args.port)


def _cmd_init(args):
    from .config import load_config
    from .reviewed_teachin import init_profile

    config = load_config(args.config)
    config['product_id'] = args.product_id

    init_profile(
        dataset_path=args.dataset,
        profile_dir=args.profile,
        config=config,
        num_images=args.num_images,
        image_list=args.images,
        reference_image=args.reference,
    )


def _cmd_regions(args):
    from .reviewed_teachin import define_regions_interactive
    define_regions_interactive(args.profile)


def _cmd_preview(args):
    from .reviewed_teachin import preview_alignment
    preview_alignment(
        profile_dir=args.profile,
        max_images=args.max_images,
        image_names=args.images,
    )


def _cmd_finalize(args):
    from .reviewed_teachin import finalize_profile
    finalize_profile(args.profile)


def _cmd_teachin(args):
    from .config import load_config
    from .teachin import run_teachin

    config = load_config(args.config)
    config['product_id'] = args.product_id
    roi = list(args.roi) if args.roi else None

    run_teachin(args.input, args.output, config, roi=roi)


def _cmd_preprocess(args):
    from .preprocess import run_preprocess

    run_preprocess(
        args.profile, args.input, args.output,
        save_debug=not args.no_debug,
    )


# ------------------------------------------------------------------

_EXAMPLES = """\
GUI Wizard (easiest)
--------------------
  python -m preprocessing_pipeline wizard

Reviewed Teach-in Workflow (CLI alternative)
--------------------------------------------
  # 1. Initialise profile and select teach-in images
  python -m preprocessing_pipeline init \\
      --dataset /path/to/VisA/pcb1/Data/Images/Normal \\
      --profile profiles/pcb1

  # 2. Define canonical crop / object mask / similarity ROI
  python -m preprocessing_pipeline regions \\
      --profile profiles/pcb1
  #    (or edit profiles/pcb1/profile.yaml directly)

  # 3. Preview alignment on teach-in images
  python -m preprocessing_pipeline preview \\
      --profile profiles/pcb1

  # 4. Finalize profile (build consensus reference)
  python -m preprocessing_pipeline finalize \\
      --profile profiles/pcb1

Batch Preprocessing
-------------------
  python -m preprocessing_pipeline preprocess \\
      --profile profiles/pcb1 \\
      --input   /path/to/VisA/pcb1/Data/Images/Normal \\
      --output  output/pcb1

  # Without per-image debug output (faster)
  python -m preprocessing_pipeline preprocess \\
      --profile profiles/pcb1 \\
      --input   /path/to/images \\
      --output  output/pcb1 \\
      --no-debug

Legacy Automatic Teach-in
-------------------------
  python -m preprocessing_pipeline teachin \\
      --input  /path/to/teachin_images \\
      --output profiles/pcb1
"""


if __name__ == '__main__':
    # Allow direct execution: python preprocessing_pipeline/pipeline.py ...
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from preprocessing_pipeline.pipeline import main as _main
    _main()
