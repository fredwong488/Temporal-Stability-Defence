"""
scripts/precompute_detections.py
---------------------------------
Run an OpenPCDet detector over a dataset split and serialise the raw
bounding-box predictions to a pickle file.  Subsequent evaluation runs can
then use --detector precomputed instead of running live GPU inference every
time.

Usage
-----
    # KITTI — PointPillars, full val split
    python scripts/precompute_detections.py \\
        --dataset kitti --detector pointpillars --split val \\
        --output data/precomputed/kitti_pointpillars_val.pkl

    # KITTI — PointRCNN, first 50 frames of val
    python scripts/precompute_detections.py \\
        --dataset kitti --detector pointrcnn \\
        --split val --num-frames 50 \\
        --output data/precomputed/kitti_pointrcnn_val50.pkl

    # NuScenes mini — PP-MultiHead, mini_val keyframes only
    python scripts/precompute_detections.py \\
        --dataset nuscenes --detector pp_multihead \\
        --nuscenes-split mini_val --keyframes-only \\
        --output data/precomputed/nuscenes_pp_multihead_mini_val.pkl

    # Override config / checkpoint (e.g. to use SECOND on KITTI)
    python scripts/precompute_detections.py \\
        --dataset kitti --detector pointpillars \\
        --config-path OpenPCDet/tools/cfgs/kitti_models/second.yaml \\
        --checkpoint-path models/openpcdet/second_7862.pth \\
        --output data/precomputed/kitti_second_val.pkl

Once saved, load the pickle via PrecomputedDetector:

    from eval_pipeline.detectors.precomputed import PrecomputedDetector
    det = PrecomputedDetector("data/precomputed/kitti_pointpillars_val.pkl")

Output format
-------------
The pickle stores:
    dict[frame_id, list[eval_pipeline.types.Prediction]]
where frame_id is a zero-padded 6-digit string for KITTI or a 16-char
sample_data token prefix for NuScenes.  All box coordinates are in the
velodyne / sensor frame (not global).

Default config + checkpoint pairs
----------------------------------
dataset=kitti,    detector=pointpillars  → pointpillar.yaml  + pointpillar_7728.pth
dataset=kitti,    detector=pointrcnn     → pointrcnn.yaml    + pointrcnn_7870.pth
dataset=nuscenes, detector=pp_multihead  → cbgs_pp_multihead.yaml + pp_multihead_nds5823_updated.pth
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DATASETS_BASE = _PROJECT_ROOT / "data" / "datasets"
_MODELS_DIR    = _PROJECT_ROOT / "models" / "openpcdet"
_CFGS_DIR      = _PROJECT_ROOT / "OpenPCDet" / "tools" / "cfgs"

DEFAULT_KITTI_ROOT       = str(_DATASETS_BASE / "KITTI")
DEFAULT_NUSCENES_ROOT    = str(_DATASETS_BASE / "nuscenes-v1.0-mini")
DEFAULT_NUSCENES_VERSION = "v1.0-mini"
DEFAULT_NUSCENES_SPLIT   = "mini_val"
DEFAULT_SCORE_THRESHOLD  = 0.3
DEFAULT_OUTPUT_DIR       = str(_PROJECT_ROOT / "data" / "precomputed")

IMAGESETS_DIR = _PROJECT_ROOT / "OpenPCDet" / "data" / "kitti" / "ImageSets"

# ---------------------------------------------------------------------------
# Detector catalogue
#
# Maps (dataset, detector_name) → (inference_class, default_config, default_ckpt)
# inference_class is one of "pointpillars" | "pointrcnn"
# ---------------------------------------------------------------------------

_DETECTOR_CATALOGUE: dict[tuple[str, str], tuple[str, str, str]] = {
    ("kitti", "pointpillars"): (
        "pointpillars",
        str(_CFGS_DIR / "kitti_models"    / "pointpillar.yaml"),
        str(_MODELS_DIR / "pointpillar_7728.pth"),
    ),
    ("kitti", "pointrcnn"): (
        "pointrcnn",
        str(_CFGS_DIR / "kitti_models"    / "pointrcnn.yaml"),
        str(_MODELS_DIR / "pointrcnn_7870.pth"),
    ),
    ("nuscenes", "pp_multihead"): (
        "pointpillars",
        str(_CFGS_DIR / "nuscenes_models" / "cbgs_pp_multihead.yaml"),
        str(_MODELS_DIR / "pp_multihead_nds5823_updated.pth"),
    ),
}

# All valid detector names (used for argparse choices)
VALID_DETECTORS = sorted({name for (_, name) in _DETECTOR_CATALOGUE})
VALID_DATASETS  = {"kitti", "nuscenes"}
KITTI_VALID_SPLITS = {"train", "val", "test"}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _resolve_detector(
    dataset: str,
    detector: str,
    config_path: str | None,
    checkpoint_path: str | None,
) -> tuple[str, str, str]:
    """Return (inference_class, resolved_config_path, resolved_checkpoint_path).

    Explicit --config-path / --checkpoint-path always win; otherwise the
    catalogue provides sensible defaults per (dataset, detector) pair.
    """
    key = (dataset, detector)
    catalogue_entry = _DETECTOR_CATALOGUE.get(key)

    if catalogue_entry is None and (config_path is None or checkpoint_path is None):
        available = [f"({d}, {det})" for (d, det) in sorted(_DETECTOR_CATALOGUE)]
        raise ValueError(
            f"No default config/checkpoint for --dataset {dataset!r} "
            f"--detector {detector!r}.\n"
            f"  Catalogue covers: {', '.join(available)}\n"
            f"  Supply --config-path and --checkpoint-path to use a custom combination."
        )

    if catalogue_entry is not None:
        cls_name, default_cfg, default_ckpt = catalogue_entry
    else:
        # Custom combo — infer inference class from detector name
        cls_name = "pointpillars" if "pillar" in detector or "multihead" in detector else "pointrcnn"

    return (
        cls_name,
        config_path    if config_path    is not None else default_cfg,
        checkpoint_path if checkpoint_path is not None else default_ckpt,
    )


def _build_detector(cls_name: str, config_path: str, checkpoint_path: str,
                    score_threshold: float, device: str):
    """Instantiate the requested OpenPCDet-backed detector."""
    if cls_name == "pointpillars":
        from eval_pipeline.detectors.pointpillars import PointPillarsDetector
        return PointPillarsDetector(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            score_threshold=score_threshold,
            device=device,
        )
    if cls_name == "pointrcnn":
        from eval_pipeline.detectors.pointrcnn import PointRCNNDetector
        return PointRCNNDetector(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            score_threshold=score_threshold,
            device=device,
        )
    raise ValueError(f"Unknown inference class: {cls_name!r}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _get_kitti_frame_ids(
    split: str,
    num_frames: int | None,
    frames: list[str] | None,
) -> list[str]:
    if frames:
        return frames
    split_file = IMAGESETS_DIR / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"KITTI split file not found: {split_file}\n"
            f"Expected OpenPCDet ImageSets at {IMAGESETS_DIR}"
        )
    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    return ids[:num_frames] if num_frames is not None else ids


def _build_dataset(args: argparse.Namespace):
    """Construct and return the appropriate dataset object."""
    if args.dataset == "kitti":
        frame_ids = _get_kitti_frame_ids(args.split, args.num_frames, args.frames)
        logging.info("KITTI split  : %s  (%d frames)", args.split if not args.frames else "custom", len(frame_ids))
        from eval_pipeline.datasets.kitti import KittiObjectDataset
        return KittiObjectDataset(root=args.kitti_root, frame_ids=frame_ids)

    # NuScenes
    logging.info(
        "NuScenes     : %s  version=%s  split=%s  keyframes_only=%s",
        args.nuscenes_root, args.nuscenes_version, args.nuscenes_split, args.keyframes_only,
    )
    from eval_pipeline.datasets.nuscenes import NuScenesDataset
    return NuScenesDataset(
        root=args.nuscenes_root,
        version=args.nuscenes_version,
        split=args.nuscenes_split,
        keyframes_only=args.keyframes_only,
    )


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

def _default_output_path(args: argparse.Namespace) -> pathlib.Path:
    """Auto-generate a descriptive output path under DEFAULT_OUTPUT_DIR."""
    if args.dataset == "kitti":
        tag = f"kitti_{args.detector}_{args.split}"
        if args.num_frames:
            tag += f"_{args.num_frames}frames"
        elif args.frames:
            tag += f"_{len(args.frames)}frames"
    else:
        tag = f"nuscenes_{args.detector}_{args.nuscenes_split}"
        if args.keyframes_only:
            tag += "_keyframes"
    return pathlib.Path(DEFAULT_OUTPUT_DIR) / f"{tag}.pkl"


# ---------------------------------------------------------------------------
# Core precomputation
# ---------------------------------------------------------------------------

def run_precompute(dataset, detector, output_path: pathlib.Path) -> dict:
    """Iterate dataset, run detector on every frame, save predictions to pickle.

    The pickle stores dict[frame_id, list[Prediction]] where Prediction is
    the eval_pipeline.types.Prediction dataclass.  All coordinates are in the
    velodyne / sensor frame.

    Returns a summary dict with num_frames and num_predictions.
    """
    from tqdm import tqdm

    predictions: dict = {}
    total_preds = 0

    for frame in tqdm(dataset, desc="Detecting", unit="frame"):
        preds = detector.predict(frame)
        predictions[frame.frame_id] = preds
        total_preds += len(preds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {"num_frames": len(predictions), "num_predictions": total_preds}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute 3D object detector predictions and cache them to a pickle "
            "file for fast replay via PrecomputedDetector."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required: dataset and detector
    parser.add_argument(
        "--dataset", required=True, choices=sorted(VALID_DATASETS),
        help="Dataset to run detection on",
    )
    parser.add_argument(
        "--detector", required=True, choices=VALID_DETECTORS,
        help=(
            "Detector to use.  KITTI: pointpillars, pointrcnn.  "
            "NuScenes: pp_multihead."
        ),
    )

    # Detector config overrides
    det_group = parser.add_argument_group("Detector overrides")
    det_group.add_argument(
        "--config-path", default=None,
        help="OpenPCDet YAML config path (overrides per-dataset catalogue default)",
    )
    det_group.add_argument(
        "--checkpoint-path", default=None,
        help="Model weights .pth path (overrides per-dataset catalogue default)",
    )
    det_group.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum detection confidence to include in the saved predictions",
    )
    det_group.add_argument(
        "--device", default="cuda:0",
        help="PyTorch device string, e.g. cuda:0 or cpu",
    )

    # Output
    parser.add_argument(
        "--output", default=None,
        help=(
            f"Destination pickle path.  Auto-generated under {DEFAULT_OUTPUT_DIR}/ "
            "when omitted (e.g. kitti_pointpillars_val.pkl)."
        ),
    )

    # KITTI-specific options
    kitti = parser.add_argument_group("KITTI options (--dataset kitti)")
    kitti.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT,
                       help="KITTI dataset root directory")
    kitti.add_argument(
        "--split", default="val", choices=sorted(KITTI_VALID_SPLITS),
        help="KITTI split — reads frame IDs from OpenPCDet/data/kitti/ImageSets/",
    )
    frame_grp = kitti.add_mutually_exclusive_group()
    frame_grp.add_argument(
        "--num-frames", type=int, default=None,
        help="Use only the first N frames from the split",
    )
    frame_grp.add_argument(
        "--frames", nargs="+", metavar="ID",
        help="Explicit zero-padded KITTI frame IDs, e.g. 000125 000070",
    )

    # NuScenes-specific options
    nusc = parser.add_argument_group("NuScenes options (--dataset nuscenes)")
    nusc.add_argument("--nuscenes-root", default=DEFAULT_NUSCENES_ROOT,
                      help="NuScenes dataset root directory")
    nusc.add_argument("--nuscenes-version", default=DEFAULT_NUSCENES_VERSION,
                      help="NuScenes version string, e.g. v1.0-mini or v1.0-trainval")
    nusc.add_argument("--nuscenes-split", default=DEFAULT_NUSCENES_SPLIT,
                      help="NuScenes split, e.g. mini_val, mini_train, val, train")
    nusc.add_argument(
        "--keyframes-only", action="store_true", default=False,
        help=(
            "Process only annotated keyframes (2 Hz) instead of all LiDAR sweeps "
            "(10 Hz).  Use this when you only need predictions at label timestamps."
        ),
    )

    args = parser.parse_args()

    # Resolve detector paths
    cls_name, config_path, checkpoint_path = _resolve_detector(
        args.dataset, args.detector, args.config_path, args.checkpoint_path,
    )

    # Resolve output path
    output_path = pathlib.Path(args.output) if args.output else _default_output_path(args)

    # Log plan
    logging.info("=" * 60)
    logging.info("Dataset      : %s", args.dataset)
    logging.info("Detector     : %s  (class=%s)", args.detector, cls_name)
    logging.info("Config       : %s", config_path)
    logging.info("Checkpoint   : %s", checkpoint_path)
    logging.info("Score thresh : %.2f", args.score_threshold)
    logging.info("Device       : %s", args.device)
    logging.info("Output       : %s", output_path)
    logging.info("=" * 60)

    # Build dataset
    dataset = _build_dataset(args)

    # Load detector (GPU-heavy — done after dataset validation)
    logging.info("Loading detector weights …")
    detector = _build_detector(cls_name, config_path, checkpoint_path,
                                args.score_threshold, args.device)

    # Run
    logging.info("Running inference on %d frames …", len(dataset))
    stats = run_precompute(dataset, detector, output_path)

    logging.info("=" * 60)
    logging.info(
        "Done: %d frames, %d predictions total (avg %.1f per frame)",
        stats["num_frames"],
        stats["num_predictions"],
        stats["num_predictions"] / max(stats["num_frames"], 1),
    )
    logging.info("Saved to: %s", output_path)
    logging.info("")
    logging.info("To use in experiments:")
    logging.info(
        "    from eval_pipeline.detectors.precomputed import PrecomputedDetector"
    )
    logging.info(
        "    det = PrecomputedDetector(%r)", str(output_path)
    )


if __name__ == "__main__":
    main()
