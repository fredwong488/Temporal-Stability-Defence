"""
tools/visualise_attack.py
-------------------------
Visualise the effect of an adversarial LiDAR attack on individual frames.

Interactively prompts for dataset, detector, attack, and frames (all
overridable via CLI flags).  For each selected frame the script:

  • applies the configured attack (with optional attack-fraction sampling)
  • runs the detector on the clean and attacked point clouds (optional)
  • renders a multi-panel figure:
      Top row    — Clean BEV | Attacked BEV
      [Iso row]  — Clean isometric | Attacked isometric  (--isometric only)
      Bottom row — Camera image with optional box overlays | Frame stats

Bounding-box overlays on the camera image project 3-D velodyne corners
through the KITTI calibration matrices.  GT boxes are always projected from
the ground-truth labels; prediction overlays require a detector.

Usage
-----
    python tools/visualise_attack.py
    python tools/visualise_attack.py --attack ora --attack-params budget=200 --show-boxes
    python tools/visualise_attack.py --dataset kitti --detector pointrcnn \\
        --attack ora --attack-params budget=100 --frames 000125 000070
    python tools/visualise_attack.py --attack ora --attack-fraction 0.5 \\
        --attack-fraction-seed 7 --attack-params budget=200 --show-boxes --isometric
    python tools/visualise_attack.py --attack ora --attack-params budget=200 \\
        --output-dir /tmp/my_vis --show-boxes
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import numpy as np
    from PIL import Image as PILImage
    from eval_pipeline.visualisation.panels import (
        draw_bev,
        draw_camera,
        draw_isometric,
        _draw_box_bev,
        _draw_box_3d,
        _project_velo_to_image,
        _project_box_to_image,
        _draw_box_image,
        _BOX_EDGES,
    )
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DATASETS_BASE = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets"
KITTI_ROOT     = f"{_DATASETS_BASE}/KITTI"
DEFAULT_NUSCENES_ROOT    = f"{_DATASETS_BASE}/nuscenes-v1.0-mini"
DEFAULT_NUSCENES_VERSION = "v1.0-mini"
DEFAULT_NUSCENES_SPLIT   = "mini_val"

IMAGESETS_DIR = _PROJECT_ROOT / "OpenPCDet" / "data" / "kitti" / "ImageSets"



# ---------------------------------------------------------------------------
# Interactive pickers
# ---------------------------------------------------------------------------

def _get_int_choice(n: int, *, allow_empty: bool = False) -> int | None:
    prompt = f"Choose [1-{n}]{' (Enter to skip)' if allow_empty else ''}: "
    while True:
        raw = input(prompt).strip()
        if allow_empty and not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw)
        print(f"  Please enter a number between 1 and {n}.")


def _pick_from_list(label: str, options: list[str], current: str | None) -> str | None:
    if current is not None:
        return current
    print(f"\n{label}:")
    print(f"  [1] (none)")
    for i, opt in enumerate(options, 2):
        print(f"  [{i}] {opt}")
    choice = _get_int_choice(len(options) + 1, allow_empty=True)
    if choice is None or choice == 1:
        return None
    return options[choice - 2]


def _get_split_frame_ids(split: str, num_frames: int | None = None) -> list[str]:
    split_file = IMAGESETS_DIR / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Expected OpenPCDet ImageSets at {IMAGESETS_DIR}"
        )
    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if num_frames is not None:
        ids = ids[:num_frames]
    return ids


def _pick_frames(frame_ids: list[str]) -> list[str]:
    """Interactively select a subset of frame IDs to visualise.

    Accepts:
      • A plain integer N     → first N frames from the split
      • Space-separated list  → frame list indices shown below (1-based)
      • 'a'                   → all frames
      • Enter                 → first 10 frames
    """
    sample = frame_ids[:min(len(frame_ids), 50)]
    print(f"\nAvailable frames (first {len(sample)} shown, {len(frame_ids)} total):")
    for i, fid in enumerate(sample, 1):
        print(f"  [{i}] {fid}")
    print()
    print("  Enter a count N to use the first N frames (e.g. '20')")
    print("  Enter space-separated list indices to pick specific frames (e.g. '1 3 7')")
    print("  Enter 'a' for all frames, or press Enter for first 10")
    while True:
        raw = input("Selection: ").strip()
        if not raw:
            return frame_ids[:10]
        if raw.lower() == "a":
            return frame_ids
        # Single integer → first-N mode
        if raw.isdigit():
            n = int(raw)
            if n < 1:
                print("  Count must be at least 1.")
                continue
            if n > len(frame_ids):
                print(f"  Only {len(frame_ids)} frames available; using all.")
                n = len(frame_ids)
            return frame_ids[:n]
        # Space-separated list of indices
        parts = raw.split()
        valid = []
        ok = True
        for p in parts:
            if p.isdigit() and 1 <= int(p) <= len(sample):
                valid.append(sample[int(p) - 1])
            else:
                print(f"  Invalid: '{p}' — must be 1–{len(sample)}.")
                ok = False
                break
        if ok and valid:
            return valid


# ---------------------------------------------------------------------------
# Stats panel
# ---------------------------------------------------------------------------

def draw_stats(
    ax: "plt.Axes",
    frame_id: str,
    is_attacked: bool,
    n_clean_preds: int,
    n_atk_preds: int | None,
    n_gt: int,
    attack_type: str | None,
    attack_params: dict,
    detector_type: str | None,
    attack_fraction: float,
) -> None:
    ax.axis("off")
    ax.set_facecolor("#f9fafb")

    lines = [
        f"Frame ID       : {frame_id}",
        f"Is attacked    : {is_attacked}",
        "",
        f"GT objects     : {n_gt}",
        f"Clean preds    : {n_clean_preds}",
        f"Attacked preds : {n_atk_preds if n_atk_preds is not None else 'N/A'}",
        "",
        f"Attack         : {attack_type or '(none)'}",
    ]
    if attack_type and attack_params:
        for k, v in attack_params.items():
            lines.append(f"  {k:<13}: {v}")
    lines += [
        f"Attack fraction: {attack_fraction:.2f}",
        "",
        f"Detector       : {detector_type or '(none)'}",
    ]

    colour = "#fee2e2" if is_attacked else "#dcfce7"
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colour, alpha=0.8),
    )
    ax.set_title("Frame stats", fontsize=8, pad=4)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def render_frame(
    frame_id: str,
    clean_lidar: np.ndarray,
    attacked_lidar: np.ndarray | None,
    clean_preds: list,
    attacked_preds: list | None,
    gt_labels: list,
    calib: Any,
    camera_image: np.ndarray | None,
    is_attacked: bool,
    attack_type: str | None,
    attack_params: dict,
    detector_type: str | None,
    attack_fraction: float,
    show_boxes: bool,
    show_isometric: bool,
    roi_min: tuple[float, float],
    roi_max: tuple[float, float],
    output_path: pathlib.Path,
    is_nuscenes: bool = False,
) -> None:
    n_rows = 3 if show_isometric else 2
    fig_height = 30 if show_isometric else 19
    fig = plt.figure(figsize=(24, fig_height))
    fig.patch.set_facecolor("#0d1117")

    # Attacked column (col 1) is 1.5× wider; camera row gets equal height to BEV rows.
    height_ratios = ([2.5, 2.5, 2.5] if show_isometric else [2.5, 2.5])
    gs = gridspec.GridSpec(n_rows, 2, figure=fig,
                           height_ratios=height_ratios,
                           width_ratios=[1, 1.5],
                           hspace=0.40, wspace=0.22)

    ax_bev_clean = fig.add_subplot(gs[0, 0])
    ax_bev_atk   = fig.add_subplot(gs[0, 1])

    if show_isometric:
        ax_iso_clean = fig.add_subplot(gs[1, 0], projection="3d")
        ax_iso_atk   = fig.add_subplot(gs[1, 1], projection="3d")
        ax_cam   = fig.add_subplot(gs[2, 0])
        ax_stats = fig.add_subplot(gs[2, 1])
    else:
        ax_cam   = fig.add_subplot(gs[1, 0])
        ax_stats = fig.add_subplot(gs[1, 1])

    atk_lidar  = attacked_lidar if attacked_lidar is not None else clean_lidar
    atk_preds  = clean_preds if not is_attacked else (attacked_preds if attacked_preds is not None else [])
    atk_note   = "" if is_attacked else "  (no attack applied)"

    draw_bev(
        ax_bev_clean, clean_lidar, clean_preds,
        gt_labels=gt_labels, show_boxes=show_boxes,
        roi_min=roi_min, roi_max=roi_max,
        title=f"Clean BEV  |  {len(clean_preds)} prediction(s)",
        is_nuscenes=is_nuscenes,
    )
    draw_bev(
        ax_bev_atk, atk_lidar, atk_preds,
        gt_labels=None, show_boxes=show_boxes,
        roi_min=roi_min, roi_max=roi_max,
        title=f"Attacked BEV  |  {len(atk_preds)} prediction(s){atk_note}",
        is_nuscenes=is_nuscenes,
    )

    if show_isometric:
        draw_isometric(
            ax_iso_clean, clean_lidar, clean_preds,
            gt_labels=gt_labels, show_boxes=show_boxes,
            roi_min=roi_min, roi_max=roi_max,
            title=f"Clean isometric  |  {len(clean_preds)} prediction(s)",
            is_nuscenes=is_nuscenes,
        )
        draw_isometric(
            ax_iso_atk, atk_lidar, atk_preds,
            gt_labels=None, show_boxes=show_boxes,
            roi_min=roi_min, roi_max=roi_max,
            title=f"Attacked isometric  |  {len(atk_preds)} prediction(s){atk_note}",
            is_nuscenes=is_nuscenes,
        )

    draw_camera(
        ax_cam, camera_image,
        gt_labels=None,
        predictions=atk_preds,
        calib=calib,
        show_boxes=show_boxes,
        is_attacked=is_attacked,
        frame_id=frame_id,
    )
    draw_stats(
        ax_stats,
        frame_id=frame_id,
        is_attacked=is_attacked,
        n_clean_preds=len(clean_preds),
        n_atk_preds=len(atk_preds) if attacked_preds is not None else None,
        n_gt=len(gt_labels),
        attack_type=attack_type,
        attack_params=attack_params,
        detector_type=detector_type,
        attack_fraction=attack_fraction,
    )

    attacked_tag = "ATTACKED" if is_attacked else "CLEAN"
    fig.suptitle(
        f"Frame {frame_id}  —  {attacked_tag}",
        fontsize=12, color="#ef4444" if is_attacked else "#22c55e",
        y=0.99,
    )

    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Camera image loading
# ---------------------------------------------------------------------------

def _load_camera_image(dataset_root: str, frame_id: str) -> np.ndarray | None:
    img_path = (
        pathlib.Path(dataset_root)
        / "data_object_image_2" / "training" / "image_2"
        / f"{frame_id}.png"
    )
    if not img_path.exists():
        return None
    try:
        return np.asarray(PILImage.open(img_path).convert("RGB"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise LiDAR attack effect: BEV, isometric, and camera views",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["kitti", "nuscenes"],
                        help="Dataset backend")
    parser.add_argument("--kitti-root", type=str, default=KITTI_ROOT)
    parser.add_argument("--kitti-split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--num-frames", type=int, default=None, metavar="N",
                        help="Use the first N frames from the split (skips interactive count prompt)")
    parser.add_argument("--nuscenes-root", type=str, default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--nuscenes-version", type=str, default=DEFAULT_NUSCENES_VERSION)
    parser.add_argument("--nuscenes-split", type=str, default=DEFAULT_NUSCENES_SPLIT)

    # Components
    parser.add_argument("--attack", type=str, default=None,
                        help="Attack type (e.g. ora).  Interactive picker if omitted.")
    parser.add_argument("--attack-params", nargs="*", metavar="KEY=VALUE", default=None,
                        help="Attack constructor kwargs as key=value pairs, "
                             "e.g. --attack-params budget=200 seed=42")
    parser.add_argument("--attack-fraction", type=float, default=1.0, metavar="F",
                        help="Fraction of frames to attack (0.0–1.0)")
    parser.add_argument("--attack-fraction-seed", type=int, default=0)
    parser.add_argument("--detector", type=str, default=None,
                        help="Detector type (e.g. pointrcnn, pointpillars).  "
                             "Interactive picker if omitted.")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Detector score threshold")

    # Frame selection
    parser.add_argument("--frames", nargs="+", metavar="ID",
                        help="Explicit KITTI frame IDs to process, e.g. 000125 000070")

    # Output
    parser.add_argument("--output-dir", type=str, default="results/attack_vis",
                        help="Directory to save rendered figures")
    parser.add_argument("--show-boxes", action="store_true", default=False,
                        help="Overlay detection and GT bounding boxes on all panels")
    parser.add_argument("--isometric", action="store_true", default=False,
                        help="Add isometric 3-D views between BEV and camera rows")

    args = parser.parse_args()

    if not HAS_MPL:
        sys.exit("matplotlib, numpy, and Pillow are required.\n"
                 "pip install matplotlib numpy Pillow")

    # -----------------------------------------------------------------------
    # Interactive selection of dataset, attack, detector
    # -----------------------------------------------------------------------

    # Dataset
    dataset_type = args.dataset
    if dataset_type is None:
        dataset_type = _pick_from_list(
            "Dataset", ["kitti", "nuscenes"], None
        ) or "kitti"
        print(f"  → {dataset_type}")

    # Attack
    attack_type = args.attack
    if attack_type is None:
        attack_type = _pick_from_list(
            "Attack type", ["ora"], None
        )
        print(f"  → {attack_type or '(none)'}")

    # Detector
    detector_type = args.detector
    if detector_type is None:
        detector_type = _pick_from_list(
            "Detector",
            ["pointrcnn", "pointpillars", "pointpillars_nuscenes"],
            None,
        )
        print(f"  → {detector_type or '(none)'}")

    # -----------------------------------------------------------------------
    # Build dataset_params
    # -----------------------------------------------------------------------
    dataset_params: dict = {}
    if dataset_type == "kitti":
        if args.frames:
            frame_ids = args.frames
        else:
            all_ids = _get_split_frame_ids(args.kitti_split)
            if args.num_frames is not None:
                frame_ids = all_ids[:args.num_frames]
            else:
                frame_ids = _pick_frames(all_ids)
        dataset_params["root"] = args.kitti_root
        dataset_params["frame_ids"] = frame_ids
    else:
        dataset_params["root"] = args.nuscenes_root
        dataset_params["version"] = args.nuscenes_version
        dataset_params["split"] = args.nuscenes_split
        if args.frames:
            dataset_params["scene_names"] = args.frames

    # -----------------------------------------------------------------------
    # Instantiate attack, detector, dataset
    # -----------------------------------------------------------------------
    from eval_pipeline.runner import _attack_registry, _detector_registry, _dataset_registry

    dataset_cls = _dataset_registry()[dataset_type]
    dataset = dataset_cls(**dataset_params)

    attack = None
    attack_params_used: dict = {}
    if attack_type is not None:
        cls = _attack_registry()[attack_type]
        for kv in (args.attack_params or []):
            if "=" not in kv:
                sys.exit(f"--attack-params: expected key=value, got '{kv}'")
            k, v = kv.split("=", 1)
            # coerce to int → float → str
            try:
                attack_params_used[k] = int(v)
            except ValueError:
                try:
                    attack_params_used[k] = float(v)
                except ValueError:
                    attack_params_used[k] = v
        attack = cls(**attack_params_used)

    detector = None
    if detector_type is not None:
        det_cls = _detector_registry()[detector_type]
        detector = det_cls(score_threshold=args.confidence_threshold)

    # -----------------------------------------------------------------------
    # RNG for attack-fraction sampling (mirrors EvalPipeline)
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(args.attack_fraction_seed)

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir.resolve()}")
    print(f"Frames to process: {len(dataset)}")

    # -----------------------------------------------------------------------
    # Per-frame processing + rendering
    # -----------------------------------------------------------------------
    try:
        from tqdm import tqdm
        _iter = tqdm(dataset, desc="Rendering", unit="frame")
    except ImportError:
        _iter = iter(dataset)

    for frame in _iter:
        # Determine whether to attack this frame
        do_attack = (attack is not None) and (rng.random() < args.attack_fraction)

        # Clean detector run
        clean_preds: list = []
        if detector is not None:
            clean_preds = detector.predict(frame)

        # Apply attack
        attacked_frame = None
        attacked_preds: list | None = None
        if do_attack:
            attacked_frame = attack.apply(frame)
            if detector is not None:
                attacked_preds = detector.predict(attacked_frame)

        attacked_lidar = attacked_frame.lidar if attacked_frame is not None else None
        is_attacked    = attacked_frame is not None

        # Camera image (KITTI only)
        camera_image: np.ndarray | None = None
        if dataset_type == "kitti":
            camera_image = _load_camera_image(args.kitti_root, frame.frame_id)

        render_frame(
            frame_id=frame.frame_id,
            clean_lidar=frame.lidar,
            attacked_lidar=attacked_lidar,
            clean_preds=clean_preds,
            attacked_preds=attacked_preds,
            gt_labels=frame.labels,
            calib=frame.kitti_calib,
            camera_image=camera_image,
            is_attacked=is_attacked,
            attack_type=attack_type,
            attack_params=attack_params_used,
            detector_type=detector_type,
            attack_fraction=args.attack_fraction,
            show_boxes=args.show_boxes,
            show_isometric=args.isometric,
            roi_min=(0.0, -5.0),
            roi_max=(30.0, 5.0),
            output_path=out_dir / f"{frame.frame_id}.png",
            is_nuscenes=(dataset_type == "nuscenes"),
        )

    print(f"\nDone. {len(dataset)} figure(s) saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
