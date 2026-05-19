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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import numpy as np
    from PIL import Image as PILImage
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
# Velodyne → camera projection
# ---------------------------------------------------------------------------

def _project_velo_to_image(
    pts_velo: np.ndarray,   # (N, 3)
    calib: Any,             # Calibration dataclass
) -> tuple[np.ndarray, np.ndarray]:
    """Project velodyne points to image pixel coordinates.

    Returns (pixels (N, 2), mask (N,)) where mask selects points with depth > 0
    that land within a finite image.
    """
    if calib is None or calib.P2 is None:
        return np.zeros((0, 2)), np.zeros(len(pts_velo), dtype=bool)

    N = len(pts_velo)
    pts_h = np.hstack([pts_velo, np.ones((N, 1), dtype=np.float32)])   # (N, 4)
    pts_cam  = (calib.Tr_velo_to_cam @ pts_h.T).T                      # (N, 3)
    pts_rect = (calib.R0_rect @ pts_cam.T).T                           # (N, 3)
    pts_rect_h = np.hstack([pts_rect, np.ones((N, 1), dtype=np.float32)])
    pts_img = (calib.P2 @ pts_rect_h.T).T                              # (N, 3)
    depth = pts_img[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(depth > 0, pts_img[:, 0] / depth, np.nan)
        v = np.where(depth > 0, pts_img[:, 1] / depth, np.nan)
    mask = (depth > 0) & np.isfinite(u) & np.isfinite(v)
    pixels = np.column_stack([u, v])
    return pixels, mask


def _project_box_to_image(
    corners_velo: np.ndarray,   # (8, 3)
    calib: Any,
) -> np.ndarray | None:
    """Project 8 box corners to image space; return (8, 2) or None."""
    pixels, mask = _project_velo_to_image(corners_velo, calib)
    if mask.sum() < 4:
        return None
    return pixels


# Edges of a 3-D bounding box defined by corner index pairs
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def _draw_box_image(
    ax: "plt.Axes",
    corners_img: np.ndarray,   # (8, 2) – may contain nan
    color: str,
    linewidth: float = 1.5,
    alpha: float = 0.9,
) -> None:
    for i, j in _BOX_EDGES:
        p1, p2 = corners_img[i], corners_img[j]
        if np.any(np.isnan([p1, p2])):
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=linewidth, alpha=alpha)


# ---------------------------------------------------------------------------
# BEV drawing helpers  (reuse logic from visualise_frames.py)
# ---------------------------------------------------------------------------

def _draw_box_bev(ax: "plt.Axes", corners_velo: np.ndarray, color: str, lw: float = 1.5) -> None:
    xy = np.asarray(corners_velo)[:, :2]
    _, idx = np.unique(np.round(xy, 3), axis=0, return_index=True)
    xy_u = xy[idx]
    c = xy_u.mean(axis=0)
    order = np.argsort(np.arctan2(xy_u[:, 1] - c[1], xy_u[:, 0] - c[0]))
    ax.add_patch(plt.Polygon(xy_u[order], closed=True, fill=False,
                              edgecolor=color, linewidth=lw))


def draw_bev(
    ax: "plt.Axes",
    lidar: np.ndarray,
    predictions: list,
    gt_labels: list | None = None,
    show_boxes: bool = True,
    roi_min: tuple[float, float] = (0.0, -5.0),
    roi_max: tuple[float, float] = (30.0, 5.0),
    title: str = "",
) -> None:
    ax.set_facecolor("#111827")

    pts = lidar
    if len(pts) > 20_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20_000, replace=False)]
    z_norm = np.clip(pts[:, 2], -3.0, 1.5)
    ax.scatter(pts[:, 0], pts[:, 1], c=z_norm, s=0.4,
               cmap="viridis", alpha=0.7, rasterized=True, zorder=1)

    ax.add_patch(mpatches.Rectangle(
        (roi_min[0], roi_min[1]),
        roi_max[0] - roi_min[0], roi_max[1] - roi_min[1],
        linewidth=1, edgecolor="white", facecolor="none",
        linestyle="--", alpha=0.4, zorder=2,
    ))

    if show_boxes:
        if gt_labels:
            for lbl in gt_labels:
                _draw_box_bev(ax, lbl.corners_velo, color="#22c55e", lw=1.2)
        for pred in predictions:
            corners = pred.corners_velo if hasattr(pred, "corners_velo") else pred["corners_velo"]
            _draw_box_bev(ax, corners, color="#60a5fa", lw=1.2)

    ax.set_xlim(-3, roi_max[0] + 3)
    ax.set_ylim(roi_min[1] - 3, roi_max[1] + 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8, color="white")
    ax.set_ylabel("y (m)", fontsize=8, color="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    ax.set_title(title, fontsize=8, color="white", pad=4)

    legend_items = []
    if show_boxes:
        if gt_labels:
            legend_items.append(mpatches.Patch(facecolor="none", edgecolor="#22c55e", label="GT"))
        if predictions:
            legend_items.append(mpatches.Patch(facecolor="none", edgecolor="#60a5fa", label="Prediction"))
    if legend_items:
        ax.legend(handles=legend_items, fontsize=6, loc="upper right",
                  facecolor="#1f2937", labelcolor="white", framealpha=0.7)


# ---------------------------------------------------------------------------
# Isometric (3-D) drawing helpers
# ---------------------------------------------------------------------------

def _draw_box_3d(ax: Any, corners_velo: np.ndarray, color: str, lw: float = 1.0) -> None:
    corners = np.asarray(corners_velo)
    z_mid = corners[:, 2].mean()
    bottom = corners[corners[:, 2] < z_mid]
    top    = corners[corners[:, 2] >= z_mid]

    def _sort_angle(pts: np.ndarray) -> np.ndarray:
        c = pts[:, :2].mean(axis=0)
        return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]

    bottom = _sort_angle(bottom)
    top    = _sort_angle(top)

    def _face(pts: np.ndarray) -> None:
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], [pts[i, 2], pts[j, 2]],
                    color=color, linewidth=lw, alpha=0.85)

    _face(bottom)
    _face(top)
    for b in bottom:
        t = top[np.argmin(np.linalg.norm(top[:, :2] - b[:2], axis=1))]
        ax.plot([b[0], t[0]], [b[1], t[1]], [b[2], t[2]], color=color, linewidth=lw, alpha=0.85)


def draw_isometric(
    ax: Any,
    lidar: np.ndarray,
    predictions: list,
    gt_labels: list | None = None,
    show_boxes: bool = True,
    roi_min: tuple[float, float] = (0.0, -5.0),
    roi_max: tuple[float, float] = (30.0, 5.0),
    title: str = "",
) -> None:
    ax.set_facecolor("#111827")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#111827")
        pane.set_edgecolor("#374151")

    pts = lidar
    if len(pts) > 15_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 15_000, replace=False)]
    z_norm = np.clip(pts[:, 2], -3.0, 1.5)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=z_norm, s=0.3,
               cmap="viridis", alpha=0.6, rasterized=True, zorder=1,
               vmin=-3.0, vmax=1.5)

    if show_boxes:
        if gt_labels:
            for lbl in gt_labels:
                _draw_box_3d(ax, lbl.corners_velo, color="#22c55e")
        for pred in predictions:
            corners = pred.corners_velo if hasattr(pred, "corners_velo") else pred["corners_velo"]
            _draw_box_3d(ax, corners, color="#60a5fa")

    x_span = roi_max[0] + 6
    y_span = roi_max[1] - roi_min[1] + 6
    ax.set_xlim(-3, roi_max[0] + 3)
    ax.set_ylim(roi_min[1] - 3, roi_max[1] + 3)
    ax.set_zlim(-3.0, 3.0)
    ax.set_box_aspect([x_span, y_span, 6.0])
    ax.set_xlabel("x (m)", fontsize=7, color="white")
    ax.set_ylabel("y (m)", fontsize=7, color="white")
    ax.set_zlabel("z (m)", fontsize=7, color="white")
    ax.tick_params(colors="white", labelsize=6)
    ax.set_title(title, fontsize=8, color="white", pad=4)
    ax.view_init(elev=25, azim=215)


# ---------------------------------------------------------------------------
# Camera image panel
# ---------------------------------------------------------------------------

def draw_camera(
    ax: "plt.Axes",
    image: np.ndarray | None,
    gt_labels: list | None,
    predictions: list,
    calib: Any,
    show_boxes: bool,
    is_attacked: bool,
    frame_id: str,
) -> None:
    ax.set_facecolor("#111827")

    if image is None:
        ax.text(0.5, 0.5, "Camera image not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#9ca3af")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

        if show_boxes and calib is not None:
            # GT boxes — green
            for lbl in (gt_labels or []):
                corners_img = _project_box_to_image(lbl.corners_velo, calib)
                if corners_img is not None:
                    _draw_box_image(ax, corners_img, color="#22c55e", linewidth=1.5)

            # Prediction boxes — blue
            for pred in predictions:
                corners_velo = pred.corners_velo if hasattr(pred, "corners_velo") else np.array(pred["corners_velo"])
                corners_img = _project_box_to_image(corners_velo, calib)
                if corners_img is not None:
                    _draw_box_image(ax, corners_img, color="#60a5fa", linewidth=1.5)

    attack_label = "ATTACKED" if is_attacked else "CLEAN"
    attack_colour = "#ef4444" if is_attacked else "#22c55e"
    ax.set_title(
        f"Camera image — frame {frame_id}   [{attack_label}]",
        fontsize=8, color=attack_colour, pad=4,
    )


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
) -> None:
    n_rows = 3 if show_isometric else 2
    fig_height = 22 if show_isometric else 14
    fig = plt.figure(figsize=(18, fig_height))
    fig.patch.set_facecolor("#0d1117")

    height_ratios = ([2.5, 2.5, 1.8] if show_isometric else [2.5, 1.8])
    gs = gridspec.GridSpec(n_rows, 2, figure=fig,
                           height_ratios=height_ratios,
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
    atk_preds  = attacked_preds if attacked_preds is not None else []
    atk_note   = "" if is_attacked else "  (no attack applied)"

    draw_bev(
        ax_bev_clean, clean_lidar, clean_preds,
        gt_labels=gt_labels, show_boxes=show_boxes,
        roi_min=roi_min, roi_max=roi_max,
        title=f"Clean BEV  |  {len(clean_preds)} prediction(s)",
    )
    draw_bev(
        ax_bev_atk, atk_lidar, atk_preds,
        gt_labels=None, show_boxes=show_boxes,
        roi_min=roi_min, roi_max=roi_max,
        title=f"Attacked BEV  |  {len(atk_preds)} prediction(s){atk_note}",
    )

    if show_isometric:
        draw_isometric(
            ax_iso_clean, clean_lidar, clean_preds,
            gt_labels=gt_labels, show_boxes=show_boxes,
            roi_min=roi_min, roi_max=roi_max,
            title=f"Clean isometric  |  {len(clean_preds)} prediction(s)",
        )
        draw_isometric(
            ax_iso_atk, atk_lidar, atk_preds,
            gt_labels=gt_labels, show_boxes=show_boxes,
            roi_min=roi_min, roi_max=roi_max,
            title=f"Attacked isometric  |  {len(atk_preds)} prediction(s){atk_note}",
        )

    draw_camera(
        ax_cam, camera_image,
        gt_labels=gt_labels,
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

    fig.savefig(output_path, dpi=110, facecolor=fig.get_facecolor())
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
        )

    print(f"\nDone. {len(dataset)} figure(s) saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
