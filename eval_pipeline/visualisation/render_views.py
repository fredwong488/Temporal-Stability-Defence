"""
eval_pipeline/visualisation/render_views.py
--------------------------------------------
Render the three LLM-defense views (BEV, isometric, camera) to in-memory
PNG bytes without writing to disk.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .panels import draw_bev, draw_camera, draw_isometric

if TYPE_CHECKING:
    from eval_pipeline.types import Frame, Prediction


def render_three_views(
    frame: "Frame",
    predictions: list["Prediction"],
    *,
    roi_min: tuple[float, float] = (0.0, -5.0),
    roi_max: tuple[float, float] = (30.0, 5.0),
    dpi: int = 150,
) -> dict[str, bytes]:
    """Render BEV, isometric, and camera panels to PNG bytes.

    GT labels are intentionally omitted — the LLM sees only the raw sensor
    data and detector predictions, not the ground truth.

    Returns a dict with keys 'bev', 'isometric', 'camera'.
    """
    lidar = frame.lidar
    image = frame.image
    calib = frame.kitti_calib
    is_nuscenes = frame.nuscenes_ego_pose is not None

    views: dict[str, bytes] = {}

    # BEV
    fig_bev, ax_bev = plt.subplots(figsize=(8, 6))
    fig_bev.patch.set_facecolor("#0d1117")
    draw_bev(
        ax_bev, lidar, predictions,
        gt_labels=None, show_boxes=True,
        roi_min=roi_min, roi_max=roi_max,
        title="LiDAR BEV",
        is_nuscenes=is_nuscenes,
    )
    buf = io.BytesIO()
    fig_bev.savefig(buf, format="png", dpi=dpi, facecolor=fig_bev.get_facecolor())
    plt.close(fig_bev)
    views["bev"] = buf.getvalue()

    # Isometric
    fig_iso = plt.figure(figsize=(9, 6))
    fig_iso.patch.set_facecolor("#0d1117")
    ax_iso = fig_iso.add_subplot(111, projection="3d")
    draw_isometric(
        ax_iso, lidar, predictions,
        gt_labels=None, show_boxes=True,
        roi_min=roi_min, roi_max=roi_max,
        title="LiDAR Isometric",
        is_nuscenes=is_nuscenes,
    )
    buf = io.BytesIO()
    fig_iso.savefig(buf, format="png", dpi=dpi, facecolor=fig_iso.get_facecolor())
    plt.close(fig_iso)
    views["isometric"] = buf.getvalue()

    # Camera
    fig_cam, ax_cam = plt.subplots(figsize=(10, 4))
    fig_cam.patch.set_facecolor("#0d1117")
    draw_camera(
        ax_cam, image,
        gt_labels=None, predictions=predictions,
        calib=calib, show_boxes=True,
        is_attacked=frame.is_attacked,
        frame_id=frame.frame_id,
    )
    buf = io.BytesIO()
    fig_cam.savefig(buf, format="png", dpi=dpi, facecolor=fig_cam.get_facecolor())
    plt.close(fig_cam)
    views["camera"] = buf.getvalue()

    return views
