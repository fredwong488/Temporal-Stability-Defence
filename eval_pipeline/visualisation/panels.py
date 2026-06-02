"""
eval_pipeline/visualisation/panels.py
--------------------------------------
Panel-drawing helpers shared between tools/visualise_attack.py and the
LLM defense renderer.  Each function accepts an existing matplotlib Axes and
mutates it in-place; callers are responsible for figure creation and saving.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise ImportError("matplotlib is required for visualisation panels") from exc


# ---------------------------------------------------------------------------
# Velodyne → camera projection
# ---------------------------------------------------------------------------

def _project_velo_to_image(
    pts_velo: np.ndarray,
    calib: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Project velodyne points to image pixel coordinates.

    Returns (pixels (N, 2), mask (N,)) where mask selects points with depth > 0
    that land within a finite image.
    """
    if calib is None or calib.P2 is None:
        return np.zeros((0, 2)), np.zeros(len(pts_velo), dtype=bool)

    N = len(pts_velo)
    pts_h = np.hstack([pts_velo, np.ones((N, 1), dtype=np.float32)])
    pts_cam = (calib.Tr_velo_to_cam @ pts_h.T).T
    pts_rect = (calib.R0_rect @ pts_cam.T).T
    pts_rect_h = np.hstack([pts_rect, np.ones((N, 1), dtype=np.float32)])
    pts_img = (calib.P2 @ pts_rect_h.T).T
    depth = pts_img[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(depth > 0, pts_img[:, 0] / depth, np.nan)
        v = np.where(depth > 0, pts_img[:, 1] / depth, np.nan)
    mask = (depth > 0) & np.isfinite(u) & np.isfinite(v)
    pixels = np.column_stack([u, v])
    return pixels, mask


def _project_box_to_image(
    corners_velo: np.ndarray,
    calib: Any,
) -> np.ndarray | None:
    """Project 8 box corners to image space; return (8, 2) or None."""
    pixels, mask = _project_velo_to_image(corners_velo, calib)
    if mask.sum() < 4:
        return None
    return pixels


# Edges of a 3-D bounding box defined by corner index pairs
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _draw_box_image(
    ax: "plt.Axes",
    corners_img: np.ndarray,
    color: str,
    linewidth: float = 1.5,
    alpha: float = 0.9,
    label: str | None = None,
) -> None:
    for i, j in _BOX_EDGES:
        p1, p2 = corners_img[i], corners_img[j]
        if np.any(np.isnan([p1, p2])):
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=linewidth, alpha=alpha)
    if label:
        valid = corners_img[~np.any(np.isnan(corners_img), axis=1)]
        if len(valid) > 0:
            ax.text(valid[:, 0].min(), valid[:, 1].min() - 2, label,
                    color=color, fontsize=6, ha="left", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _rotate_pts_cw90(pts: np.ndarray) -> np.ndarray:
    """Return a copy of (N, 4+) lidar with x/y rotated 90° CW: new_x=y, new_y=-x."""
    out = pts.copy()
    out[:, 0], out[:, 1] = pts[:, 1].copy(), -pts[:, 0].copy()
    return out


def _rotate_corners_cw90(corners: np.ndarray) -> np.ndarray:
    """Return a copy of (8, 3) corners with x/y rotated 90° CW."""
    out = np.asarray(corners, dtype=float).copy()
    out[:, 0], out[:, 1] = corners[:, 1].copy(), -corners[:, 0].copy()
    return out



# ---------------------------------------------------------------------------
# BEV drawing helpers
# ---------------------------------------------------------------------------

def _draw_box_bev(
    ax: "plt.Axes",
    corners_velo: np.ndarray,
    color: str,
    lw: float = 1.5,
    label: str | None = None,
) -> None:
    xy = np.asarray(corners_velo)[:, :2]
    _, idx = np.unique(np.round(xy, 3), axis=0, return_index=True)
    xy_u = xy[idx]
    c = xy_u.mean(axis=0)
    order = np.argsort(np.arctan2(xy_u[:, 1] - c[1], xy_u[:, 0] - c[0]))
    ax.add_patch(plt.Polygon(xy_u[order], closed=True, fill=False,
                              edgecolor=color, linewidth=lw))
    if label:
        ax.text(c[0], c[1], label, color=color, fontsize=5.5,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#0d1117", alpha=0.6, edgecolor="none"))


def draw_bev(
    ax: "plt.Axes",
    lidar: np.ndarray,
    predictions: list,
    gt_labels: list | None = None,
    show_boxes: bool = True,
    roi_min: tuple[float, float] = (0.0, -5.0),
    roi_max: tuple[float, float] = (30.0, 5.0),
    title: str = "",
    is_nuscenes: bool = False,
) -> None:
    if is_nuscenes:
        lidar = _rotate_pts_cw90(lidar)
        if gt_labels is not None:
            gt_labels = [dataclasses.replace(l, corners_velo=_rotate_corners_cw90(l.corners_velo)) for l in gt_labels]
        predictions = [
            (dataclasses.replace(p, corners_velo=_rotate_corners_cw90(p.corners_velo)) if hasattr(p, "corners_velo")
             else {**p, "corners_velo": _rotate_corners_cw90(p["corners_velo"])})
            for p in predictions
        ]

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
                _draw_box_bev(ax, lbl.corners_velo, color="#22c55e", lw=1.2, label=lbl.type)
        for pred in predictions:
            corners = pred.corners_velo if hasattr(pred, "corners_velo") else pred["corners_velo"]
            pred_type = pred.type if hasattr(pred, "type") else pred.get("type", "")
            pred_score = pred.score if hasattr(pred, "score") else pred.get("score", None)
            pred_label = f"{pred_type} {pred_score:.2f}" if pred_score is not None else pred_type
            _draw_box_bev(ax, corners, color="#60a5fa", lw=1.2, label=pred_label)

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
    top = corners[corners[:, 2] >= z_mid]

    def _sort_angle(pts: np.ndarray) -> np.ndarray:
        c = pts[:, :2].mean(axis=0)
        return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]

    bottom = _sort_angle(bottom)
    top = _sort_angle(top)

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
    is_nuscenes: bool = False,
) -> None:
    if is_nuscenes:
        lidar = _rotate_pts_cw90(lidar)
        if gt_labels is not None:
            gt_labels = [dataclasses.replace(l, corners_velo=_rotate_corners_cw90(l.corners_velo)) for l in gt_labels]
        predictions = [
            (dataclasses.replace(p, corners_velo=_rotate_corners_cw90(p.corners_velo)) if hasattr(p, "corners_velo")
             else {**p, "corners_velo": _rotate_corners_cw90(p["corners_velo"])})
            for p in predictions
        ]

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
# Camera panel
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

        h, w = image.shape[:2]

        if show_boxes and calib is not None:
            for lbl in (gt_labels or []):
                corners_img = _project_box_to_image(lbl.corners_velo, calib)
                if corners_img is not None:
                    _draw_box_image(ax, corners_img, color="#22c55e", linewidth=0.8, label=lbl.type)

            for pred in predictions:
                corners_velo = pred.corners_velo if hasattr(pred, "corners_velo") else np.array(pred["corners_velo"])
                corners_img = _project_box_to_image(corners_velo, calib)
                if corners_img is not None:
                    pred_type = pred.type if hasattr(pred, "type") else pred.get("type", "")
                    pred_score = pred.score if hasattr(pred, "score") else pred.get("score", None)
                    pred_label = f"{pred_type} {pred_score:.2f}" if pred_score is not None else pred_type
                    _draw_box_image(ax, corners_img, color="#60a5fa", linewidth=0.8, label=pred_label)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    attack_label = "ATTACKED" if is_attacked else "CLEAN"
    attack_colour = "#ef4444" if is_attacked else "#22c55e"
    ax.set_title(
        f"Camera image — frame {frame_id}   [{attack_label}]",
        fontsize=8, color=attack_colour, pad=4,
    )
