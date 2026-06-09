"""
defenses/tc2/defense.py
-----------------------
TC2Defense — integrates 3D-TC2 temporal consistency checking into the
eval_pipeline's BaseDefense interface.

The defense reads past frames from either pipeline history deque
(clean by default, dirty if history_source="dirty") and:
  1. Transforms each past sweep's LiDAR into the current sensor frame
     via each Frame's nuscenes_ego_pose.
  2. Subsamples by frame_skip and voxelises into a BEV occupancy stack.
  3. Runs MotionNet to predict future displacement fields.
  4. Checks whether the detector's boxes in the current frame are consistent
     with the motion forecast (TC2 consistency rule).
  5. Returns is_attack_detected=True if any in-ROI box is inconsistent.

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

import logging
import time
from collections import deque
from typing import Literal

import numpy as np
import torch

from ...base import BaseDefense
from ...types import DetectionResult, Frame, FrameHistory

logger = logging.getLogger(__name__)


class TC2Defense(BaseDefense):
    """Temporal Consistency Check defense using MotionNet.

    Parameters
    ----------
    model_path
        Path to the MotionNet pretrained checkpoint (.pth).
    net
        Which MotionNet variant to use: ``"MotionNet"`` or ``"MotionNetMGDA"``.
    device
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    nsweeps_back
        Number of past sweeps to feed into MotionNet (default 20).
    frame_skip
        Subsampling factor: take every (frame_skip+1)-th sweep.
        With nsweeps_back=20 and frame_skip=3, 5 sweeps are used.
    voxel_size
        (vx, vy, vz) BEV cell dimensions in metres.
    bev_extents
        ((xmin,xmax),(ymin,ymax),(zmin,zmax)) BEV coverage in metres.
    roi_x
        (x_min, x_max) sensor-frame gate on box centre x.
    roi_y
        (y_min, y_max) sensor-frame gate on box centre y.
    static_norm_threshold
        Displacement L2-norm below which a cell is treated as static.
    use_adj_frame_pred
        Accumulate adjacent-frame displacement predictions (as in TC2 paper).
    use_motion_state_masking
        Zero out displacement fields where MotionNet predicts static motion.
    target_classes
        Only evaluate consistency for boxes of these detection_name classes.
    history_source
        ``"clean"``  — use pre-attack history (default; matches TC2 paper's
                        benign-history threat model).
        ``"dirty"`` — use post-attack history (for stress-testing temporal
                        attacks that also poison past sweeps).
    """

    def __init__(
        self,
        model_path: str = "models/merl/motionnet_MGDA.pth",
        net: str = "MotionNetMGDA",
        device: str = "cuda",
        nsweeps_back: int = 20,
        frame_skip: int = 3,
        voxel_size: tuple[float, float, float] = (0.25, 0.25, 0.4),
        bev_extents: tuple[tuple[float, float], ...] = (
            (-32.0, 32.0), (-32.0, 32.0), (-3.0, 2.0)
        ),
        roi_x: tuple[float, float] = (-8.0, 8.0),
        roi_y: tuple[float, float] = (8.0, 30.0),
        static_norm_threshold: float = 0.4,
        use_adj_frame_pred: bool = True,
        use_motion_state_masking: bool = True,
        target_classes: tuple[str, ...] = ("car",),
        history_source: Literal["clean", "dirty"] = "clean",
    ) -> None:
        self.model_path = model_path
        self.net = net
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.nsweeps_back = nsweeps_back
        self.frame_skip = frame_skip
        self.voxel_size = voxel_size
        self.bev_extents = bev_extents
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.static_norm_threshold = static_norm_threshold
        self.use_adj_frame_pred = use_adj_frame_pred
        self.use_motion_state_masking = use_motion_state_masking
        self.target_classes = tuple(target_classes)
        self.history_source = history_source

        self._model = None  # lazy-loaded on first call

    @property
    def temporal_window(self) -> int:
        return self.nsweeps_back + 1  # current + nsweeps_back prior frames

    # ------------------------------------------------------------------
    # BaseDefense contract
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        hist: deque[Frame] = history.clean if self.history_source == "clean" else history.dirty

        if len(hist) < self.nsweeps_back:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={
                    "reason": "cold_start",
                    "n_history": len(hist),
                    "nsweeps_required": self.nsweeps_back,
                },
            )

        if frame.nuscenes_ego_pose is None:
            raise ValueError(
                "TC2Defense requires frame.nuscenes_ego_pose to be set. "
                "Use NuScenesDataset (not KittiObjectDataset) with this defense."
            )

        if not frame.predictions:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"reason": "no_predictions"},
            )

        _t0_total = time.perf_counter()

        # Build past-sweep list in the current sensor frame
        _t0 = time.perf_counter()
        sweep_lidar = self._prepare_sweeps(frame, hist)
        _elapsed_prepare_sweeps_s = time.perf_counter() - _t0

        # Voxelise and run MotionNet
        _t0 = time.perf_counter()
        bev_tensor, non_empty_map = self._build_bev(sweep_lidar)
        _elapsed_build_bev_s = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        disp_pred, cat_pred_raw, motion_pred = self._run_model(bev_tensor)
        _elapsed_run_model_s = time.perf_counter() - _t0

        # Optional masking (same as TC2.py:705-723)
        if self.use_adj_frame_pred:
            for c in range(1, disp_pred.shape[0]):
                disp_pred[c] = disp_pred[c] + disp_pred[c - 1]

        if self.use_motion_state_masking:
            motion_pred_np = motion_pred.cpu().numpy()
            motion_pred_np = np.argmax(motion_pred_np, axis=1)  # (1, H, W) — axis 1 = motion-cat dim
            motion_mask = motion_pred_np == 0                    # static cells

            # cat_pred_raw is (5, H, W) after squeeze(0) in _run_model; axis=0 selects over categories.
            cat_np = np.argmax(cat_pred_raw.cpu().numpy(), axis=0)  # (H, W)
            cat_mask = (cat_np == 0) & (non_empty_map == 1)         # background occupied cells

            weight_map = np.ones_like(motion_pred_np, dtype=np.float32)  # (1, H, W)
            weight_map[motion_mask] = 0.0
            weight_map[:, cat_mask] = 0.0  # cat_mask is (H, W); index with leading slice
            weight_map = weight_map[:, :, :, np.newaxis]  # (1, H, W, 1)
            disp_pred = disp_pred * weight_map

        # Use the prediction step that corresponds to t=0 (current time).
        # MotionNet's reference is the most recent past sweep at t = -(frame_skip+1).
        # After use_adj_frame_pred accumulation, disp_pred[i] is the cumulative
        # displacement from the reference to t_ref + (i+1) sweeps.
        # For t=0: t_ref + (i+1) = 0  →  i = frame_skip.
        last_disp = disp_pred[self.frame_skip]  # (H, W, 2)
        cat_pred_logits = cat_pred_raw.cpu().numpy()  # (5, H, W)

        from .core import run_tc2_check
        _t0 = time.perf_counter()
        result = run_tc2_check(
            disp_pred=last_disp,
            cat_pred_logits=cat_pred_logits,
            non_empty_map=non_empty_map,
            predictions=frame.predictions,
            voxel_size=self.voxel_size,
            bev_extents=self.bev_extents,
            roi_x=self.roi_x,
            roi_y=self.roi_y,
            static_norm_threshold=self.static_norm_threshold,
            target_classes=self.target_classes,
        )

        _elapsed_tc2_check_s = time.perf_counter() - _t0

        is_attack = result.n_inconsistent > 0
        confidence = (result.n_inconsistent / result.n_boxes_in_roi
                      if result.n_boxes_in_roi > 0 else 0.0)

        return DetectionResult(
            is_attack_detected=is_attack,
            confidence=float(confidence),
            metadata={
                "n_boxes_in_roi": result.n_boxes_in_roi,
                "n_inconsistent": result.n_inconsistent,
                "history_source": self.history_source,
                "elapsed_s": {
                    "prepare_sweeps": _elapsed_prepare_sweeps_s,
                    "build_bev": _elapsed_build_bev_s,
                    "run_model": _elapsed_run_model_s,
                    "tc2_check": _elapsed_tc2_check_s,
                    "total": time.perf_counter() - _t0_total,
                },
                "per_box": [
                    {
                        "type": r.box_type,
                        "is_consistent": r.is_consistent,
                        "dominant_cell_cat": r.dominant_cell_cat,
                        "n_cells_per_cat": r.n_cells_per_cat,
                    }
                    for r in result.box_results
                ],
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_sweeps(self, current_frame: Frame, hist: deque[Frame]) -> list[np.ndarray]:
        """Return a list of past point clouds transformed into the current sensor frame.

        Takes the most recent nsweeps_back frames from hist, subsamples by
        frame_skip. The current frame is intentionally excluded: MotionNet's
        reference is the most recent past sweep (t = -frame_skip-1 sweeps), and
        disp_pred[frame_skip] then corresponds to the prediction for t=0
        (current time), which is compared against current detector boxes.
        """
        past_frames = list(hist)[-self.nsweeps_back:]  # most recent nsweeps_back, oldest first
        # Subsample past frames: indices 0, frame_skip+1, 2*(frame_skip+1), ... (oldest first)
        indices = list(range(0, len(past_frames), self.frame_skip + 1))
        selected = [past_frames[i] for i in indices]

        cur_ego_inv = np.linalg.inv(current_frame.nuscenes_ego_pose.astype(np.float64))
        sweep_lidar: list[np.ndarray] = []

        for f in selected:
            if f.nuscenes_ego_pose is None:
                logger.warning(
                    "TC2Defense: past frame %s has no nuscenes_ego_pose — skipping", f.frame_id
                )
                continue
            # Transform: past-sensor → global → current-sensor
            T = cur_ego_inv @ f.nuscenes_ego_pose.astype(np.float64)
            pts_xyz = f.lidar[:, :3].astype(np.float64)  # (N, 3)
            pts_transformed = (T[:3, :3] @ pts_xyz.T + T[:3, 3:4]).T.astype(np.float32)
            sweep_lidar.append(np.hstack([pts_transformed, f.lidar[:, 3:4]]))  # keep intensity

        return sweep_lidar

    def _build_bev(self, sweeps: list[np.ndarray]) -> tuple[torch.Tensor, np.ndarray]:
        from .core import build_bev_stack
        return build_bev_stack(sweeps, self.voxel_size, self.bev_extents)

    def _run_model(
        self, bev_tensor: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Load model (lazy) and run forward pass.

        Returns
        -------
        disp_pred   : (n_future, H, W, 2) numpy array
        cat_pred    : (5, H, W) float tensor (raw logits)
        motion_pred : (1, 2, H, W) float tensor (motion state logits)
        """
        if self._model is None:
            self._model = self._load_model()

        bev_tensor = bev_tensor.to(self.device)

        with torch.no_grad():
            if self.net == "MotionNet":
                model = self._model
                model.eval()
                disp_t, cat_t, motion_t = model(bev_tensor)
            else:
                enc, head = self._model
                enc.eval(); head.eval()
                feats = enc(bev_tensor)
                disp_t, cat_t, motion_t = head(feats)

        # disp_t: (1*n_future, 2, H, W) → (n_future, H, W, 2)
        disp_np = disp_t.cpu().numpy()
        disp_np = np.transpose(disp_np, (0, 2, 3, 1))

        # cat_t: (1, 5, H, W) → squeeze batch → (5, H, W)
        cat_t = cat_t.squeeze(0)  # keep on device for masking step

        return disp_np, cat_t, motion_t

    @staticmethod
    def _strip_module_prefix(state_dict: dict) -> dict:
        """Remove 'module.' prefix added by DataParallel when saving checkpoints."""
        if all(k.startswith("module.") for k in state_dict):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def _load_model(self):
        from .motionnet import FeatEncoder, MotionNet, MotionNetMGDA
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if self.net == "MotionNet":
            model = MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=13)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(self._strip_module_prefix(state))
            model.to(self.device)
            logger.info("TC2Defense: loaded MotionNet from %s", self.model_path)
            return model
        elif self.net == "MotionNetMGDA":
            encoder = FeatEncoder(height_feat_size=13)
            head = MotionNetMGDA(out_seq_len=20, motion_category_num=2)
            enc_state = checkpoint.get("encoder_state_dict", checkpoint)
            head_state = checkpoint.get("head_state_dict", checkpoint)
            encoder.load_state_dict(self._strip_module_prefix(enc_state))
            head.load_state_dict(self._strip_module_prefix(head_state))
            encoder.to(self.device); head.to(self.device)
            logger.info("TC2Defense: loaded MotionNetMGDA from %s", self.model_path)
            return encoder, head
        else:
            raise ValueError(f"Unknown net '{self.net}'. Choose 'MotionNet' or 'MotionNetMGDA'.")
