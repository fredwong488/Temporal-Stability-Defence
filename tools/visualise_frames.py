"""
tools/visualise_frames.py
-------------------------
Interactive CLI to visualise per-frame attack / defense results produced by
scripts/run_sweep.py --save-frames.

Works with any defense (or no defense at all).  The bottom-left panel
auto-detects what to show:
  • VoidRegionDefense  → occupancy grid with shadow-cluster colouring
  • any other defense  → generic key/value dump of defense metadata
  • no defense         → "No defense metadata" placeholder

Layout per frame (2 × 2 grid, or 3 × 2 with --isometric):
  Top-left:     Clean BEV   — lidar z-coloured, clean predictions, optional GT
  Top-right:    Attacked BEV — lidar z-coloured, attacked predictions,
                               obstacle cluster AABBs / centroids, optional GT
                               (lidar shown is clean — attacked lidar not stored)
  [Iso row]:    Clean / attacked isometric 3-D views (--isometric only)
  Bottom-left:  Defense-specific panel (occupancy grid or metadata dump)
  Bottom-right: Frame statistics

Usage
-----
    python tools/visualise_frames.py
    python tools/visualise_frames.py --results-dir /path/to/results
    python tools/visualise_frames.py --run 2026-04-23-12-00-00
    python tools/visualise_frames.py --run 2026-04-23-12-00-00 --experiment ora_budget_40
    python tools/visualise_frames.py --run 2026-04-23-12-00-00 --filter fp
    python tools/visualise_frames.py --isometric
"""

from __future__ import annotations

import argparse
import json
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
    from tqdm import tqdm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_RESULTS_DIR = "results"
VALID_FILTERS = {"all", "tp", "tn", "fp", "fn"}

# Keys that identify void-region defense metadata
_VOID_REGION_KEY = "empty_cell_positions"

# Metadata keys already rendered verbatim in draw_stats — omit from generic dump
_STATS_SHOWN_META_KEYS = {
    "n_empty_cells", "n_clusters", "n_obstacle_clusters",
    "obstacle_centroids", "obstacle_cluster_aabbs", "obstacle_cluster_sizes",
    "obstacle_matches_gt", "empty_cell_positions", "empty_cell_cluster_labels",
    "reason", "cluster_details",
}


# ---------------------------------------------------------------------------
# Interactive pickers (mirror visualise_metrics.py pattern)
# ---------------------------------------------------------------------------

def _get_int_choice(n: int, *, allow_empty: bool = False) -> int | None:
    """Prompt until the user enters a valid integer in [1, n]. Returns None if allow_empty and input is blank."""
    prompt = f"Choose [1-{n}]{' (Enter to skip)' if allow_empty else ''}: "
    while True:
        raw = input(prompt).strip()
        if allow_empty and not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw)
        print(f"  Please enter a number between 1 and {n}.")


def list_run_dirs(results_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return subdirectories that contain at least one *_frames.jsonl file."""
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and any(d.glob("*_frames.jsonl"))
    )


def load_run_metadata(run_dir: pathlib.Path) -> dict:
    """Read summary info from the first experiment JSON in the run directory."""
    json_files = sorted(
        p for p in run_dir.glob("*.json")
        if not p.stem.endswith("_frames")
    )
    if not json_files:
        return {}
    try:
        with open(json_files[0]) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        return {
            "num_frames":    data.get("num_frames"),
            "attack_type":   cfg.get("attack_type"),
            "detector_type": cfg.get("detector_type"),
            "defense_type":  cfg.get("defense_type"),
        }
    except Exception:
        return {}


def pick_run_dir(results_dir: pathlib.Path, run_name: str | None) -> pathlib.Path:
    dirs = list_run_dirs(results_dir)
    if not dirs:
        sys.exit(f"No result directories found under '{results_dir}'.\n"
                 "Re-run experiments with --save-frames to generate per-frame data.")

    if run_name:
        matches = [d for d in dirs if d.name == run_name]
        if not matches:
            sys.exit(f"Run '{run_name}' not found. Available: {[d.name for d in dirs]}")
        return matches[0]

    print(f"\nAvailable run directories in '{results_dir}':")
    for i, d in enumerate(dirs, 1):
        n_experiments = len(list(d.glob("*_frames.jsonl")))
        meta = load_run_metadata(d)
        parts = []
        if meta.get("attack_type"):
            parts.append(meta["attack_type"])
        if meta.get("detector_type"):
            parts.append(meta["detector_type"])
        if meta.get("defense_type"):
            parts.append(meta["defense_type"])
        if meta.get("num_frames") is not None:
            parts.append(f"{meta['num_frames']} frames")
        meta_str = f"  [{', '.join(parts)}]" if parts else ""
        print(f"  [{i}] {d.name}  ({n_experiments} experiment(s)){meta_str}")

    print()
    return dirs[_get_int_choice(len(dirs)) - 1]


def pick_experiment(run_dir: pathlib.Path, experiment_name: str | None) -> list[str]:
    """Return a list of experiment stems to render.

    Returns all experiments if the user selects the 'all' option, or a
    single-element list otherwise.  The --experiment CLI arg always returns
    a single-element list.
    """
    jsonl_files = sorted(run_dir.glob("*_frames.jsonl"))
    names = [p.stem.removesuffix("_frames") for p in jsonl_files]

    if not names:
        sys.exit(f"No *_frames.jsonl files found in {run_dir}.")

    if experiment_name:
        if experiment_name not in names:
            sys.exit(f"Experiment '{experiment_name}' not found. Available: {names}")
        return [experiment_name]

    print(f"\nAvailable experiments in '{run_dir.name}':")
    print(f"  [1] all experiments")
    for i, name in enumerate(names, 1):
        results_path = run_dir / f"{name}.json"
        n_frames, n_attacked, n_detected = "?", "?", "?"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    data = json.load(f)
                de = data.get("defense_effectiveness", {})
                n_frames = data.get("num_frames", "?")
                tp = de.get("tp", 0)
                fn = de.get("fn", 0)
                fp = de.get("fp", 0)
                tn = de.get("tn", 0)
                n_attacked = tp + fn
                n_detected = tp + fp
            except Exception:
                pass
        print(f"  [{i + 1}] {name}  (frames={n_frames}, attacked={n_attacked}, detected={n_detected})")

    print()
    choice = _get_int_choice(len(names) + 1)
    return names if choice == 1 else [names[choice - 2]]


def pick_filter(current: str) -> str:
    if current != "all":
        return current
    options = ["all", "tp", "tn", "fp", "fn"]
    labels = [
        "all  — every frame",
        "tp   — attacked and detected (true positives)",
        "tn   — clean and not detected (true negatives)",
        "fp   — clean but detected (false positives)",
        "fn   — attacked but not detected (false negatives)",
    ]
    print("\nFilter frames by outcome:")
    for i, lbl in enumerate(labels, 1):
        print(f"  [{i}] {lbl}")
    choice = _get_int_choice(len(options), allow_empty=True)
    return options[choice - 1] if choice is not None else "all"


def _frame_outcome(frame_data: dict) -> str:
    dr = frame_data.get("defense_result") or {}
    detected = dr.get("is_attack_detected", False)
    attacked = frame_data.get("is_attacked", False)
    if detected and attacked:
        return "tp"
    if not detected and not attacked:
        return "tn"
    if detected and not attacked:
        return "fp"
    return "fn"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment(run_dir: pathlib.Path, experiment_name: str) -> tuple[dict, list[dict]]:
    results_path = run_dir / f"{experiment_name}.json"
    frames_path  = run_dir / f"{experiment_name}_frames.jsonl"

    with open(results_path) as f:
        results = json.load(f)
    frames = []
    with open(frames_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return results, frames


def open_kitti_dataset(kitti_root: str, frame_ids: list[str]):
    """Instantiate KittiObjectDataset once for the given frame IDs (lazy iterator)."""
    from eval_pipeline.datasets.kitti import KittiObjectDataset
    return KittiObjectDataset(root=kitti_root, frame_ids=frame_ids)


# ---------------------------------------------------------------------------
# Drawing helpers — 3-D
# ---------------------------------------------------------------------------

def _draw_box_3d(ax: Any, corners_velo: Any, color: str, linewidth: float = 1.0) -> None:
    corners = np.asarray(corners_velo)  # (8, 3)
    z_mid  = corners[:, 2].mean()
    bottom = corners[corners[:, 2] <  z_mid]
    top    = corners[corners[:, 2] >= z_mid]

    def _sort_by_angle(pts: np.ndarray) -> np.ndarray:
        c = pts[:, :2].mean(axis=0)
        return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]

    bottom = _sort_by_angle(bottom)
    top    = _sort_by_angle(top)

    def _draw_face(pts: np.ndarray) -> None:
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            ax.plot([pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    [pts[i, 2], pts[j, 2]],
                    color=color, linewidth=linewidth, alpha=0.85)

    _draw_face(bottom)
    _draw_face(top)

    # Vertical edges: pair each bottom corner with its nearest top corner by XY
    for b in bottom:
        t = top[np.argmin(np.linalg.norm(top[:, :2] - b[:2], axis=1))]
        ax.plot([b[0], t[0]], [b[1], t[1]], [b[2], t[2]],
                color=color, linewidth=linewidth, alpha=0.85)


def _draw_aabb_3d(
    ax: Any,
    aabb_min: list[float],
    aabb_max: list[float],
    color: str,
    linewidth: float = 1.5,
    alpha: float = 0.6,
) -> None:
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    edges = [
        ((x0,y0,z0),(x1,y0,z0)), ((x1,y0,z0),(x1,y1,z0)),
        ((x1,y1,z0),(x0,y1,z0)), ((x0,y1,z0),(x0,y0,z0)),
        ((x0,y0,z1),(x1,y0,z1)), ((x1,y0,z1),(x1,y1,z1)),
        ((x1,y1,z1),(x0,y1,z1)), ((x0,y1,z1),(x0,y0,z1)),
        ((x0,y0,z0),(x0,y0,z1)), ((x1,y0,z0),(x1,y0,z1)),
        ((x1,y1,z0),(x1,y1,z1)), ((x0,y1,z0),(x0,y1,z1)),
    ]
    for a, b in edges:
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]],
                color=color, linewidth=linewidth, alpha=alpha)


def draw_isometric(
    ax: Any,
    lidar: "np.ndarray",
    predictions: list[dict],
    gt_labels: Any = None,
    show_gt: bool = True,
    obstacle_aabbs: list | None = None,
    obstacle_centroids: list | None = None,
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
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=z_norm,
               s=0.3, cmap="viridis", alpha=0.6, rasterized=True, zorder=1,
               vmin=-3.0, vmax=1.5)

    if show_gt and gt_labels:
        for label in gt_labels:
            _draw_box_3d(ax, label.corners_velo, color="#22c55e", linewidth=1.0)

    for pred in predictions:
        _draw_box_3d(ax, pred["corners_velo"], color="#60a5fa", linewidth=1.0)

    if obstacle_aabbs:
        for aabb in obstacle_aabbs:
            _draw_aabb_3d(ax, aabb[0], aabb[1], color="#ef4444")
    elif obstacle_centroids:
        for c in obstacle_centroids:
            ax.scatter([c[0]], [c[1]], [c[2]], marker="x",
                       color="#ef4444", s=80, zorder=5, depthshade=False)

    x_span = roi_max[0] + 6
    y_span = roi_max[1] - roi_min[1] + 6
    z_span = 6.0
    ax.set_xlim(-3, roi_max[0] + 3)
    ax.set_ylim(roi_min[1] - 3, roi_max[1] + 3)
    ax.set_zlim(-3.0, 3.0)
    ax.set_box_aspect([x_span, y_span, z_span])
    ax.set_xlabel("x (m)", fontsize=7, color="white")
    ax.set_ylabel("y (m)", fontsize=7, color="white")
    ax.set_zlabel("z (m)", fontsize=7, color="white")
    ax.tick_params(colors="white", labelsize=6)
    ax.set_title(title, fontsize=8, color="white", pad=4)
    ax.view_init(elev=25, azim=215)

    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401 — ensures 3d is registered


# ---------------------------------------------------------------------------
# Drawing helpers — BEV
# ---------------------------------------------------------------------------

def _draw_box_bev(ax: "plt.Axes", corners_velo: Any, color: str, linewidth: float = 1.5) -> None:
    xy = np.asarray(corners_velo)[:, :2]
    _, idx = np.unique(np.round(xy, 3), axis=0, return_index=True)
    xy_u = xy[idx]
    c = xy_u.mean(axis=0)
    order = np.argsort(np.arctan2(xy_u[:, 1] - c[1], xy_u[:, 0] - c[0]))
    poly = plt.Polygon(
        xy_u[order], closed=True, fill=False,
        edgecolor=color, linewidth=linewidth,
    )
    ax.add_patch(poly)


def _draw_aabb_bev(
    ax: "plt.Axes",
    aabb_min: list[float],
    aabb_max: list[float],
    color: str,
    linewidth: float = 1.5,
    alpha: float = 0.25,
) -> None:
    x0, y0 = aabb_min[0], aabb_min[1]
    w = aabb_max[0] - aabb_min[0]
    h = aabb_max[1] - aabb_min[1]
    rect = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="square,pad=0",
        linewidth=linewidth, edgecolor=color,
        facecolor=color, alpha=alpha,
    )
    ax.add_patch(rect)


def draw_bev(
    ax: "plt.Axes",
    lidar: "np.ndarray",
    predictions: list[dict],
    gt_labels: Any = None,
    show_gt: bool = True,
    obstacle_aabbs: list | None = None,
    obstacle_centroids: list | None = None,
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

    roi_rect = mpatches.Rectangle(
        (roi_min[0], roi_min[1]),
        roi_max[0] - roi_min[0], roi_max[1] - roi_min[1],
        linewidth=1, edgecolor="white", facecolor="none",
        linestyle="--", alpha=0.4, zorder=2,
    )
    ax.add_patch(roi_rect)

    if show_gt and gt_labels:
        for label in gt_labels:
            _draw_box_bev(ax, label.corners_velo, color="#22c55e", linewidth=1.2)

    for pred in predictions:
        _draw_box_bev(ax, pred["corners_velo"], color="#60a5fa", linewidth=1.2)

    if obstacle_aabbs:
        for aabb in obstacle_aabbs:
            _draw_aabb_bev(ax, aabb[0], aabb[1], color="#ef4444")
    elif obstacle_centroids:
        for c in obstacle_centroids:
            ax.plot(c[0], c[1], "x", color="#ef4444",
                    markersize=10, markeredgewidth=2, zorder=5)

    ax.set_xlim(-3, roi_max[0] + 3)
    ax.set_ylim(roi_min[1] - 3, roi_max[1] + 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8, color="white")
    ax.set_ylabel("y (m)", fontsize=8, color="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    ax.set_title(title, fontsize=8, color="white", pad=4)

    legend_items = [mpatches.Patch(facecolor="none", edgecolor="#60a5fa", label="Prediction")]
    if show_gt and gt_labels:
        legend_items.insert(0, mpatches.Patch(facecolor="none", edgecolor="#22c55e", label="GT"))
    if obstacle_aabbs or obstacle_centroids:
        legend_items.append(mpatches.Patch(facecolor="#ef4444", alpha=0.4, label="Obstacle cluster"))
    ax.legend(handles=legend_items, fontsize=6, loc="upper right",
              facecolor="#1f2937", labelcolor="white", framealpha=0.7)


# ---------------------------------------------------------------------------
# Drawing helpers — bottom-left panel
# ---------------------------------------------------------------------------

def draw_occupancy_grid(
    ax: "plt.Axes",
    metadata: dict,
    roi_min: tuple[float, float] = (0.0, -5.0),
    roi_max: tuple[float, float] = (30.0, 5.0),
) -> None:
    """VoidRegionDefense-specific panel: empty cells coloured by shadow-cluster label."""
    ax.set_facecolor("#f9fafb")

    positions      = metadata.get("empty_cell_positions", [])
    cluster_labels = metadata.get("empty_cell_cluster_labels", [])
    positions      = np.array(positions)
    cluster_labels = np.array(cluster_labels)
    n_clusters     = metadata.get("n_clusters", 0)

    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax.scatter(positions[noise_mask, 0], positions[noise_mask, 1],
                   c="#d1d5db", s=3, alpha=0.5, zorder=1, label="empty (noise)")

    _tab10 = matplotlib.colormaps["tab10"].colors
    for i in range(n_clusters):
        mask = cluster_labels == i
        if mask.any():
            ax.scatter(positions[mask, 0], positions[mask, 1],
                       c=[_tab10[i % len(_tab10)]], s=8, zorder=2, label=f"shadow {i}")

    ax.add_patch(mpatches.Rectangle(
        (roi_min[0], roi_min[1]),
        roi_max[0] - roi_min[0], roi_max[1] - roi_min[1],
        linewidth=1, edgecolor="#374151", facecolor="none", linestyle="--", zorder=3,
    ))

    ax.set_xlim(roi_min[0] - 1, roi_max[0] + 1)
    ax.set_ylim(roi_min[1] - 1, roi_max[1] + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"Occupancy Grid — {len(positions)} empty cells, {n_clusters} shadow clusters",
        fontsize=8, pad=4,
    )
    if n_clusters > 0:
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)


def draw_defense_metadata(ax: "plt.Axes", metadata: dict) -> None:
    """Generic bottom-left panel: key/value dump of defense metadata."""
    ax.axis("off")
    ax.set_facecolor("#f9fafb")

    if not metadata:
        ax.text(0.5, 0.5, "No defense metadata",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title("Defense Metadata", fontsize=8, pad=4)
        return

    lines: list[str] = []
    for k, v in metadata.items():
        if k in _STATS_SHOWN_META_KEYS:
            continue
        if isinstance(v, list):
            lines.append(f"{k}: [{len(v)} items]")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in list(v.items())[:6]:
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")

    if not lines:
        lines = ["(all fields shown in stats panel)"]

    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f9ff", alpha=0.8),
    )
    ax.set_title("Defense Metadata", fontsize=8, pad=4)


def _draw_bottom_left(
    ax: "plt.Axes",
    metadata: dict,
    roi_min: tuple[float, float],
    roi_max: tuple[float, float],
) -> None:
    """Dispatch to the appropriate bottom-left panel based on metadata content."""
    if _VOID_REGION_KEY in metadata:
        draw_occupancy_grid(ax, metadata, roi_min=roi_min, roi_max=roi_max)
    else:
        draw_defense_metadata(ax, metadata)


# ---------------------------------------------------------------------------
# Drawing helpers — stats panel
# ---------------------------------------------------------------------------

def draw_stats(ax: "plt.Axes", frame_data: dict) -> None:
    ax.axis("off")
    ax.set_facecolor("#f9fafb")

    dr       = frame_data.get("defense_result") or {}
    meta     = dr.get("metadata", {})
    detected = dr.get("is_attack_detected", "N/A")
    attacked = frame_data.get("is_attacked", False)
    outcome  = _frame_outcome(frame_data).upper()

    lines = [
        f"Frame ID         : {frame_data['frame_id']}",
        f"Is attacked      : {attacked}",
        f"Attack detected  : {detected}",
        f"Outcome          : {outcome}",
        f"Confidence       : {dr.get('confidence', 'N/A')}",
        "",
        f"Clean preds      : {len(frame_data.get('clean_predictions') or [])}",
        f"Attacked preds   : {len(frame_data.get('attacked_predictions') or [])}",
    ]

    # Void-region specific block — only shown when those keys are present
    void_keys = {"n_empty_cells", "n_clusters", "n_obstacle_clusters"}
    if any(k in meta for k in void_keys):
        lines += [
            "",
            f"Empty cells      : {meta.get('n_empty_cells', 'N/A')}",
            f"Shadow clusters  : {meta.get('n_clusters', 'N/A')}",
            f"Obstacle clusters: {meta.get('n_obstacle_clusters', 'N/A')}",
        ]
    else:
        # Generic: show scalar metadata fields not already in the dedicated panel
        scalars = [
            (k, v) for k, v in meta.items()
            if k not in _STATS_SHOWN_META_KEYS and not isinstance(v, (list, dict))
        ]
        if scalars:
            lines.append("")
            for k, v in scalars[:8]:
                lines.append(f"{k:<17}: {v}")

    if meta.get("obstacle_centroids"):
        lines += ["", "Obstacle centroids (x, y, z):"]
        for i, c in enumerate(meta["obstacle_centroids"][:5]):
            lines.append(f"  [{i}] ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")

    colour = "#dcfce7" if outcome in ("TP", "TN") else "#fee2e2"
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colour, alpha=0.8),
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def render_frame(
    frame_data: dict,
    kitti_frame: Any,
    show_gt: bool,
    show_isometric: bool,
    roi_min: tuple[float, float],
    roi_max: tuple[float, float],
    output_path: pathlib.Path,
) -> None:
    fig_height = 20 if show_isometric else 12
    fig = plt.figure(figsize=(18, fig_height))
    fig.patch.set_facecolor("#0d1117")

    if show_isometric:
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               height_ratios=[2.5, 2.5, 1.2],
                               hspace=0.42, wspace=0.22)
    else:
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[2.5, 1.2],
                               hspace=0.38, wspace=0.22)

    ax_clean = fig.add_subplot(gs[0, 0])
    ax_atk   = fig.add_subplot(gs[0, 1])
    if show_isometric:
        ax_iso_clean = fig.add_subplot(gs[1, 0], projection="3d")
        ax_iso_atk   = fig.add_subplot(gs[1, 1], projection="3d")
        ax_grid  = fig.add_subplot(gs[2, 0])
        ax_stats = fig.add_subplot(gs[2, 1])
    else:
        ax_grid  = fig.add_subplot(gs[1, 0])
        ax_stats = fig.add_subplot(gs[1, 1])

    lidar       = kitti_frame.lidar
    dr          = frame_data.get("defense_result") or {}
    meta        = dr.get("metadata", {})
    is_attacked = frame_data.get("is_attacked", False)
    clean_preds = frame_data.get("clean_predictions") or []
    raw_atk_preds = frame_data.get("attacked_predictions")
    atk_preds   = raw_atk_preds if raw_atk_preds is not None else clean_preds

    obstacle_aabbs = meta.get("obstacle_cluster_aabbs") or None
    obstacle_centroids = (
        meta.get("obstacle_centroids")
        if not meta.get("obstacle_cluster_aabbs") else None
    )
    atk_title = (
        f"{'Attacked' if is_attacked else 'Clean (no attack)'} BEV"
        f"  |  {len(atk_preds)} prediction(s)"
        f"  |  detected={'yes' if dr.get('is_attack_detected') else 'no'}"
        + ("\n(lidar shown is clean — attacked lidar not stored)" if is_attacked else "")
    )

    draw_bev(
        ax_clean, lidar, clean_preds,
        gt_labels=kitti_frame.labels if show_gt else None,
        show_gt=show_gt,
        roi_min=roi_min, roi_max=roi_max,
        title=f"Clean BEV  |  {len(clean_preds)} prediction(s)",
    )
    draw_bev(
        ax_atk, lidar, atk_preds,
        gt_labels=kitti_frame.labels if show_gt else None,
        show_gt=show_gt,
        obstacle_aabbs=obstacle_aabbs,
        obstacle_centroids=obstacle_centroids,
        roi_min=roi_min, roi_max=roi_max,
        title=atk_title,
    )

    if show_isometric:
        draw_isometric(
            ax_iso_clean, lidar, clean_preds,
            gt_labels=kitti_frame.labels if show_gt else None,
            show_gt=show_gt,
            roi_min=roi_min, roi_max=roi_max,
            title=f"Clean isometric  |  {len(clean_preds)} prediction(s)",
        )
        draw_isometric(
            ax_iso_atk, lidar, atk_preds,
            gt_labels=kitti_frame.labels if show_gt else None,
            show_gt=show_gt,
            obstacle_aabbs=obstacle_aabbs,
            obstacle_centroids=obstacle_centroids,
            roi_min=roi_min, roi_max=roi_max,
            title=atk_title.replace("BEV", "isometric"),
        )

    _draw_bottom_left(ax_grid, meta, roi_min=roi_min, roi_max=roi_max)
    draw_stats(ax_stats, frame_data)

    fig.savefig(output_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise per-frame attack / defense results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Base results directory")
    parser.add_argument("--run", default=None, metavar="NAME",
                        help="Run directory name to skip the interactive picker")
    parser.add_argument("--experiment", default=None, metavar="NAME",
                        help="Experiment name to skip the experiment picker")
    parser.add_argument("--filter", default="all",
                        choices=sorted(VALID_FILTERS), metavar="OUTCOME",
                        help="Only render frames matching this outcome: "
                             "all | tp | tn | fp | fn")
    parser.add_argument("--show-gt", action="store_true", default=False,
                        help="Overlay ground-truth bounding boxes")
    parser.add_argument("--isometric", action="store_true", default=False,
                        help="Add isometric 3D views below the BEV panels")
    parser.add_argument("--kitti-root", default=None, metavar="PATH",
                        help="Override the KITTI dataset root from the experiment config")
    parser.add_argument("--backend", default="matplotlib",
                        choices=["matplotlib", "plotly"],
                        help="Rendering backend (plotly not yet implemented)")
    args = parser.parse_args()

    if not HAS_MPL:
        sys.exit("matplotlib and numpy are required.\npip install matplotlib numpy")

    if args.backend == "plotly":
        raise NotImplementedError("Plotly backend not yet implemented.")

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Results directory not found: {results_dir.resolve()}")

    run_dir = pick_run_dir(results_dir, args.run)
    print(f"\nSelected run: {run_dir.name}")

    experiment_names = pick_experiment(run_dir, args.experiment)
    if len(experiment_names) > 1:
        print(f"Selected experiments: all ({len(experiment_names)})")
    else:
        print(f"Selected experiment: {experiment_names[0]}")

    frame_filter = pick_filter(args.filter)
    print(f"Frame filter: {frame_filter}")

    total_saved = 0
    for experiment_name in experiment_names:
        if len(experiment_names) > 1:
            print(f"\n--- {experiment_name} ---")

        results, frames = load_experiment(run_dir, experiment_name)
        config = results.get("config", {})
        kitti_root = args.kitti_root or config.get("kitti_root", "data/datasets/KITTI")

        # ROI: check defense_params first, then top-level config, then defaults
        defense_params = config.get("defense_params", {})
        roi_min = tuple(
            defense_params.get("roi_min")
            or config.get("roi_min")
            or [0.0, -5.0]
        )
        roi_max = tuple(
            defense_params.get("roi_max")
            or config.get("roi_max")
            or [30.0, 5.0]
        )

        if frame_filter != "all":
            frames = [f for f in frames if _frame_outcome(f) == frame_filter]
            print(f"  {len(frames)} frame(s) match filter '{frame_filter}'")
            if not frames:
                print("  Nothing to render.")
                continue

        vis_dir = run_dir / f"{experiment_name}_vis"
        if frame_filter != "all":
            vis_dir = vis_dir / frame_filter
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to {vis_dir}")

        frame_ids = [f["frame_id"] for f in frames]
        try:
            dataset = open_kitti_dataset(kitti_root, frame_ids)
        except Exception as e:
            sys.exit(f"Failed to open KITTI dataset at '{kitti_root}': {e}")

        for frame_data, kitti_frame in tqdm(zip(frames, dataset), total=len(frames), desc="Rendering", unit="frame"):
            frame_id = frame_data["frame_id"]
            render_frame(
                frame_data, kitti_frame,
                show_gt=args.show_gt,
                show_isometric=args.isometric,
                roi_min=roi_min, roi_max=roi_max,
                output_path=vis_dir / f"{frame_id}.png",
            )
        total_saved += len(frames)

    print(f"\nDone. {total_saved} figure(s) saved.")


if __name__ == "__main__":
    main()
