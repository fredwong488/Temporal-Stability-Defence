"""
defenses/tc2/core.py
--------------------
Pure-function implementation of the 3D-TC2 temporal-consistency check.

Extracted and refactored from TC2.py:vis_model_per_sample_data (lines 627-954).
All matplotlib visualisation and module-level globals have been removed.
The function is side-effect-free and returns structured results suitable for
the eval_pipeline's DetectionResult interface.

Reference
---------
You, C., Hau, Z., & Demetriou, S. (2021).
Temporal Consistency Checks to Detect LiDAR Spoofing Attacks on Autonomous
Vehicle Perception.

@inproceedings{You_2021, series={MobiSys ’21},
   title={Temporal Consistency Checks to Detect LiDAR Spoofing Attacks on Autonomous Vehicle Perception},
   url={http://dx.doi.org/10.1145/3469261.3469406},
   DOI={10.1145/3469261.3469406},
   booktitle={Proceedings of the 1st Workshop on Security and Privacy for Mobile AI},
   publisher={ACM},
   author={You, Chengzeng and Hau, Zhongyuan and Demetriou, Soteris},
   year={2021},
   month=June, pages={13–18},
   collection={MobiSys ’21} }

"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import torch

from ...types import Prediction
from .box_utils import points_in_box_bev

# MotionNet outputs 5 BEV categories (0-indexed):
#   0: background  1: bus/car  2: pedestrian  3: bicycle
#   4: other (truck, trailer, traffic_cone, barrier, …)
_CAT_NAMES = {0: "bg", 1: "bus_car", 2: "pedestrian", 3: "bicycle", 4: "other"}

# Map NuScenes detection_name → TC2 category index (0-indexed).
# A box whose class has no mapping is skipped in the consistency check.
_DET_NAME_TO_CAT_ID: dict[str, int] = {
    "car": 0,
    "bus": 0,
    "truck": 3,
    "trailer": 3,
    "construction_vehicle": 3,
    "motorcycle": 2,
    "bicycle": 2,
    "pedestrian": 1,
    "traffic_cone": 3,
    "barrier": 3,
}


@dataclasses.dataclass
class BoxCheckResult:
    """Per-box result from the TC2 consistency check."""
    box_type: str
    is_consistent: bool           # True → TC2 considers this detection plausible
    dominant_cell_cat: int        # argmax of predicted cells in box (0-indexed)
    n_cells_per_cat: list[int]    # cell counts by TC2 category


@dataclasses.dataclass
class TC2CheckResult:
    """Per-frame result from the TC2 consistency check."""
    box_results: list[BoxCheckResult]
    n_boxes_in_roi: int
    n_inconsistent: int


def run_tc2_check(
    disp_pred: np.ndarray,           # (H, W, 2) — predicted displacement field (last future step)
    cat_pred_logits: np.ndarray,     # (5, H, W) — raw category logits from MotionNet
    non_empty_map: np.ndarray,       # (H, W)    — 1.0 where BEV cells are occupied
    predictions: Sequence[Prediction],
    voxel_size: tuple[float, float, float] = (0.25, 0.25, 0.4),
    bev_extents: tuple[tuple[float, float], ...] = ((-32.0, 32.0), (-32.0, 32.0), (-3.0, 2.0)),
    roi_x: tuple[float, float] = (-8.0, 8.0),
    roi_y: tuple[float, float] = (8.0, 30.0),
    static_norm_threshold: float = 0.4,
    target_classes: tuple[str, ...] = ("car",),
) -> TC2CheckResult:
    """Run the TC2 per-box consistency check on one frame.

    Algorithm (from TC2.py:805-907):
    For each detector box that falls in the ROI and matches a target class:
      1. Compute predicted future positions of all BEV cells:
           X_cell = X_pred + U_pred,  Y_cell = Y_pred + V_pred
      2. For each MotionNet category k (0..4), count how many advected cells
         of category k fall inside the BEV footprint of the box.
      3. The category with the most cells inside is the "dominant" category.
      4. If dominant_cat == box_cat_id + 1 (1-indexed TC2 mapping), the box is
         flagged consistent; otherwise inconsistent → likely spoofed.

    Parameters
    ----------
    disp_pred        : Predicted displacement field for the last future step.
    cat_pred_logits  : Raw category logits (5, H, W) from MotionNet.
    non_empty_map    : Binary occupancy map (H, W).
    predictions      : Detector predictions for the current (possibly attacked) frame.
    voxel_size       : (vx, vy, vz) BEV voxel dimensions in metres.
    bev_extents      : ((xmin,xmax),(ymin,ymax),(zmin,zmax)) BEV coverage.
    roi_x            : (x_min, x_max) sensor-frame ROI gate on box centre.
    roi_y            : (y_min, y_max) sensor-frame ROI gate on box centre.
    static_norm_threshold : Displacement vectors below this L2-norm are zeroed.
    target_classes   : Only check boxes whose type appears in this tuple.

    Returns
    -------
    TC2CheckResult with per-box verdicts and summary counts.
    """
    H, W = disp_pred.shape[:2]
    min_pixel_x = bev_extents[0][0] / voxel_size[0]  # typically -128
    min_pixel_y = bev_extents[1][0] / voxel_size[1]  # typically -128

    # Argmax category map — non-empty cells only, 1-indexed to match TC2.
    # TC2.py additionally applies a filter_mask (max_prob == 1.0) derived from
    # ground-truth pixel_cat_map_gt to restrict evaluation to one-hot-confident cells.
    # That mask requires GT annotations and cannot be replicated at inference time,
    # so all occupied cells participate here. This increases cell counts for all
    # categories but does not change the dominant-category comparison logic.
    cat_pred_argmax = np.argmax(cat_pred_logits, axis=0) + 1  # (H, W), values 1..5
    cat_pred_argmax = (cat_pred_argmax * (non_empty_map > 0)).astype(np.int32)

    # Zero static displacement vectors
    field_pred = disp_pred.copy()  # (H, W, 2)
    field_norm = np.linalg.norm(field_pred, ord=2, axis=-1)
    field_pred[field_norm <= static_norm_threshold] = 0.0

    # Grid index arrays
    idx_x = np.arange(H)
    idx_y = np.arange(W)
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing="ij")  # (H, W)

    box_results: list[BoxCheckResult] = []
    n_in_roi = 0

    for pred in predictions:
        if pred.type not in target_classes:
            continue

        # ROI gate on box centre (sensor frame)
        if not (roi_x[0] <= pred.x <= roi_x[1] and roi_y[0] <= pred.y <= roi_y[1]):
            continue

        n_in_roi += 1

        # TC2 category id for this box (0-indexed), used for comparison below
        box_cat_id = _DET_NAME_TO_CAT_ID.get(pred.type)
        if box_cat_id is None:
            continue

        n_cells_per_cat = [0] * 5

        for k in range(5):
            # Select cells predicted as category k+1
            mask_pred = cat_pred_argmax == (k + 1)
            if not np.any(mask_pred):
                continue

            # Advected positions of these cells
            X_pred = (idx_x[mask_pred] + min_pixel_x) * voxel_size[0]
            Y_pred = (idx_y[mask_pred] + min_pixel_y) * voxel_size[1]
            U_pred = field_pred[:, :, 0][mask_pred]
            V_pred = field_pred[:, :, 1][mask_pred]
            X_cell = X_pred + U_pred
            Y_cell = Y_pred + V_pred

            points_xy = np.stack([X_cell, Y_cell], axis=0)  # (2, N)
            in_box = points_in_box_bev(pred.corners_velo, points_xy)
            n_cells_per_cat[k] = int(np.sum(in_box))

        dominant_cat = int(np.argmax(n_cells_per_cat))  # 0-indexed

        # TC2 consistency rule: dominant cat (0-indexed) == box_cat_id + 1 - 1 = box_cat_id
        # Wait — TC2.py:896: max_cell_cat_id == box_cat_id + 1
        # where max_cell_cat_id is 0-indexed (argmax of 5 categories) and
        # box_cat_id is from check_box_cat_id which maps {car→0, ped→1, bike→2}.
        # TC2 cat 1 = bus_car, TC2 cat 2 = ped, TC2 cat 3 = bike (1-indexed in model).
        # In 0-indexed model cats: 0=bg, 1=bus_car, 2=ped, 3=bike, 4=other.
        # box_cat_id for car=0 → expects dominant_cat == 0+1 = 1 → TC2 category "bus_car"
        # box_cat_id for ped=1 → expects dominant_cat == 1+1 = 2 → TC2 category "ped"
        # box_cat_id for bike=2 → expects dominant_cat == 2+1 = 3 → TC2 category "bike"
        # So the TC2 _DET_NAME_TO_CAT_ID above should map: car→0, pedestrian→1, bicycle→2
        expected_dominant = box_cat_id + 1  # as in TC2.py:896
        is_consistent = (dominant_cat == expected_dominant)

        box_results.append(BoxCheckResult(
            box_type=pred.type,
            is_consistent=is_consistent,
            dominant_cell_cat=dominant_cat,
            n_cells_per_cat=n_cells_per_cat,
        ))

    n_inconsistent = sum(1 for r in box_results if not r.is_consistent)

    return TC2CheckResult(
        box_results=box_results,
        n_boxes_in_roi=n_in_roi,
        n_inconsistent=n_inconsistent,
    )


def build_bev_stack(
    sweep_lidar_list: list[np.ndarray],
    voxel_size: tuple[float, float, float] = (0.25, 0.25, 0.4),
    bev_extents: tuple[tuple[float, float], ...] = ((-32.0, 32.0), (-32.0, 32.0), (-3.0, 2.0)),
) -> tuple[torch.Tensor, np.ndarray]:
    """Voxelise a list of point clouds into a stacked BEV occupancy tensor.

    Parameters
    ----------
    sweep_lidar_list : list of (N, 3+) arrays, each in the current sensor frame,
                       ordered oldest-first (index 0 = most distant past).
    voxel_size       : (vx, vy, vz) BEV cell sizes in metres.
    bev_extents      : ((xmin,xmax),(ymin,ymax),(zmin,zmax)).

    Returns
    -------
    bev_tensor  : float32 tensor (1, T, H, W, Z) — batch=1, T sweeps.
    non_empty   : (H, W) binary float32 occupancy map of the most recent sweep.
    """
    from .data_utils import voxelize_occupy

    extents = np.array([[bev_extents[0][0], bev_extents[0][1]],
                        [bev_extents[1][0], bev_extents[1][1]],
                        [bev_extents[2][0], bev_extents[2][1]]], dtype=np.float32)
    vsize = np.array(voxel_size, dtype=np.float32)

    # Expected grid shape from extents
    nx = int(round((bev_extents[0][1] - bev_extents[0][0]) / voxel_size[0]))  # 256
    ny = int(round((bev_extents[1][1] - bev_extents[1][0]) / voxel_size[1]))  # 256
    nz = int(round((bev_extents[2][1] - bev_extents[2][0]) / voxel_size[2]))  # 13

    frames = []
    for pts in sweep_lidar_list:
        vox = voxelize_occupy(pts.astype(np.float32), vsize, extents)
        # Pad or crop to (nx, ny, nz) if the voxeliser returns a slightly different size
        # (can happen at exact boundary due to float rounding)
        out = np.zeros((nx, ny, nz), dtype=np.float32)
        sx, sy, sz = min(vox.shape[0], nx), min(vox.shape[1], ny), min(vox.shape[2], nz)
        out[:sx, :sy, :sz] = vox[:sx, :sy, :sz]
        frames.append(out)

    # Stack: (T, H, W, Z) → unsqueeze batch → (1, T, H, W, Z)
    stacked = np.stack(frames, axis=0)  # (T, H, W, Z)
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, T, H, W, Z)

    # Non-empty map from the most recent sweep (last in the list = current frame)
    non_empty = (frames[-1].sum(axis=2) > 0).astype(np.float32)  # (H, W)

    return tensor, non_empty
