"""
tools/explore_nuscenes_frame.py
--------------------------------
Interactively explore a NuScenes keyframe: LiDAR point cloud (plotly 3-D) +
front-camera image (shown in browser).  Useful for confirming axis conventions.

Usage:
    pixi run python tools/explore_nuscenes_frame.py
    pixi run python tools/explore_nuscenes_frame.py --scene 1 --sample 2
    pixi run python tools/explore_nuscenes_frame.py --patchworkpp

    # Load attacked lidar from a precomputed shelve cache:
    pixi run python tools/explore_nuscenes_frame.py \\
        --frame-id <frame_id> --cache /path/to/cache --attacked
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATAROOT = "data/datasets/nuscenes-v1.0-mini"
VERSION  = "v1.0-mini"


def make_transform(translation, rotation_wxyz):
    q = Quaternion(rotation_wxyz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = q.rotation_matrix
    T[:3, 3]  = translation
    return T


def sensor_to_global(nusc, sd):
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return (make_transform(ep["translation"], ep["rotation"])
            @ make_transform(cs["translation"], cs["rotation"]))


def global_to_sensor(nusc, sd):
    return np.linalg.inv(sensor_to_global(nusc, sd))


# Edges of a 3-D bounding box defined by corner index pairs.
# corners_velo layout (OpenPCDet convention):
#   0-3 bottom face (z low), 4-7 top face (z high), i+4 is directly above i.
_BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # bottom face
    (4,5),(5,6),(6,7),(7,4),   # top face
    (0,4),(1,5),(2,6),(3,7),   # verticals
]


def _bbox_traces(preds, label: str, color: str, T_s2e=None):
    """Return a list of Scatter3d traces (one per box) for a prediction list."""
    import plotly.graph_objects as go
    traces = []
    for pred in preds:
        corners = pred.corners_velo.copy()   # (8, 3) sensor frame
        if T_s2e is not None:
            ones = np.ones((8, 1))
            corners = (T_s2e @ np.hstack([corners, ones]).T).T[:, :3]
        xs, ys, zs = [], [], []
        for a, b in _BOX_EDGES:
            xs += [corners[a, 0], corners[b, 0], None]
            ys += [corners[a, 1], corners[b, 1], None]
            zs += [corners[a, 2], corners[b, 2], None]
        cx, cy, cz = corners.mean(axis=0)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=2),
            name=label,
            legendgroup=label,
            showlegend=len(traces) == 0,
            hoverinfo="skip",
        ))
        traces.append(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="text",
            text=[f"{pred.type} {pred.score:.2f}"],
            textfont=dict(color=color, size=9),
            name=label,
            legendgroup=label,
            showlegend=False,
            hovertemplate=(
                f"{pred.type} score={pred.score:.2f}<br>"
                f"x={cx:.1f} y={cy:.1f} z={cz:.1f}<extra></extra>"
            ),
        ))
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default=DATAROOT)
    parser.add_argument("--version", default=VERSION,
                        help="NuScenes version string (default: v1.0-mini)")
    parser.add_argument("--scene",  type=int, default=0,
                        help="Scene index (0-based) within the split")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample (keyframe) index within the scene")
    parser.add_argument("--frame-id",
                        help="frame_id (first 16 chars of sample_data token) — "
                             "overrides --scene/--sample")
    parser.add_argument("--max-pts", type=int, default=30_000,
                        help="Downsample point cloud to this many points for speed")
    parser.add_argument("--ego", action="store_true",
                        help="Plot in ego-vehicle frame instead of LiDAR sensor frame")
    parser.add_argument("--patchworkpp", action="store_true",
                        help="Run Patchwork++ ground segmentation; colour ground grey, "
                             "non-ground by z (viridis)")
    parser.add_argument("--sensor-height", type=float, default=-1.84,
                        help="Sensor height above ground (m) passed to Patchwork++ "
                             "(default: 1.84, NuScenes LIDAR_TOP)")
    parser.add_argument("--cache",
                        help="Path stem of a shelve cache produced by EvalPipeline "
                             "(e.g. /path/to/cache — without .db/.dir extension)")
    args = parser.parse_args()

    if args.cache and not args.frame_id:
        raise SystemExit("--cache requires --frame-id")

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if args.frame_id:
        # Locate the sample_data record whose token starts with frame_id
        prefix = args.frame_id
        matches = [sd for sd in nusc.sample_data
                   if sd["token"].startswith(prefix) and sd["channel"] == "LIDAR_TOP"]
        if not matches:
            raise SystemExit(f"No sample_data token found with prefix '{prefix}'")
        if len(matches) > 1:
            raise SystemExit(f"Ambiguous frame_id '{prefix}' matched {len(matches)} tokens")
        lidar_sd = matches[0]
        if not lidar_sd["is_key_frame"]:
            prev_sample = nusc.get("sample", lidar_sd["sample_token"])
            prev_kf_sd  = nusc.get("sample_data", prev_sample["data"]["LIDAR_TOP"])
            if prev_sample["next"]:
                next_sample = nusc.get("sample", prev_sample["next"])
                next_kf_sd  = nusc.get("sample_data", next_sample["data"]["LIDAR_TOP"])
                ts = lidar_sd["timestamp"]
                nearest_kf_sd = (prev_kf_sd
                                 if abs(prev_kf_sd["timestamp"] - ts) <= abs(next_kf_sd["timestamp"] - ts)
                                 else next_kf_sd)
            else:
                nearest_kf_sd = prev_kf_sd
            print(f"Warning: this frame_id is not a keyframe — annotations will be empty. "
                  f"Nearest keyframe frame_id: {nearest_kf_sd['token'][:16]}")
        sample = nusc.get("sample", lidar_sd["sample_token"]) if lidar_sd["is_key_frame"] else None
        scene_token = nusc.get("sample", lidar_sd["sample_token"])["scene_token"]
        scene = nusc.get("scene", scene_token)
        print(f"Scene  : {scene['name']}")
        print(f"frame_id: {prefix}")
    else:
        # Navigate to the requested scene / sample
        scene = nusc.scene[args.scene]
        sample_token = scene["first_sample_token"]
        for _ in range(args.sample):
            s = nusc.get("sample", sample_token)
            if s["next"] == "":
                print(f"Scene only has {_ + 1} sample(s); using last one.")
                break
            sample_token = s["next"]
        sample = nusc.get("sample", sample_token)
        lidar_sd = None

    if lidar_sd is None:
        print(f"Scene  : {scene['name']}")
        print(f"Sample : {sample['token'][:8]}…")
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    # ------------------------------------------------------------------ #
    # Shared transforms
    # ------------------------------------------------------------------ #
    T_g2s = global_to_sensor(nusc, lidar_sd)
    ep    = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    T_g2e = np.linalg.inv(make_transform(ep["translation"], ep["rotation"]))

    # ------------------------------------------------------------------ #
    # LiDAR point cloud(s) in sensor frame
    # ------------------------------------------------------------------ #
    entry = None
    attacked_pts = None
    if args.cache:
        import shelve
        with shelve.open(args.cache, flag='r') as shelf:
            entry = shelf.get(args.frame_id)
        if entry is None:
            raise SystemExit(f"frame_id '{args.frame_id}' not found in cache '{args.cache}'")
        if entry.attacked_lidar is not None:
            attacked_pts = entry.attacked_lidar.astype(np.float32)
            print(f"Loaded attacked lidar from cache: {len(attacked_pts):,} points  "
                  f"(is_attacked={entry.is_attacked})")
        else:
            print(f"Cache found but no attacked_lidar (is_attacked={entry.is_attacked})")

    pts_path = os.path.join(args.dataroot, lidar_sd["filename"])
    pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)[:, :4]  # x,y,z,intensity

    def _subsample(p):
        if len(p) > args.max_pts:
            rng = np.random.default_rng(0)
            return p[rng.choice(len(p), args.max_pts, replace=False)]
        return p

    def _to_ego(p, T_s2e):
        ones = np.ones((len(p), 1), dtype=np.float64)
        return (T_s2e @ np.hstack([p[:, :3].astype(np.float64), ones]).T).T[:, :3]

    T_s2e = (T_g2e @ sensor_to_global(nusc, lidar_sd)) if args.ego else None

    pts = _subsample(pts)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if T_s2e is not None:
        x, y, z = _to_ego(pts, T_s2e).T

    print(f"Clean  : {len(pts):,}  |  x [{x.min():.1f}, {x.max():.1f}]"
          f"  y [{y.min():.1f}, {y.max():.1f}]  z [{z.min():.1f}, {z.max():.1f}]")

    if attacked_pts is not None:
        attacked_pts = _subsample(attacked_pts)
        ax, ay, az = attacked_pts[:, 0], attacked_pts[:, 1], attacked_pts[:, 2]
        if T_s2e is not None:
            ax, ay, az = _to_ego(attacked_pts, T_s2e).T
        print(f"Attacked: {len(attacked_pts):,}  |  x [{ax.min():.1f}, {ax.max():.1f}]"
              f"  y [{ay.min():.1f}, {ay.max():.1f}]  z [{az.min():.1f}, {az.max():.1f}]")

    # ------------------------------------------------------------------ #
    # GT annotations
    # ------------------------------------------------------------------ #

    ann_xs, ann_ys, ann_zs, ann_labels = [], [], [], []
    for ann_token in (sample["anns"] if sample is not None else []):
        ann = nusc.get("sample_annotation", ann_token)
        c   = np.array(ann["translation"] + [1.0])
        cs  = T_g2s @ c          # centroid in sensor frame
        ce  = T_g2e @ c          # centroid in ego frame (x=forward, y=left)
        cat = ann["category_name"].split(".")[1] if "." in ann["category_name"] else ann["category_name"]
        label = (f"{cat}  sensor(x={cs[0]:+.1f} y={cs[1]:+.1f})"
                 f"  ego(x={ce[0]:+.1f} y={ce[1]:+.1f})")
        cx, cy, cz = (ce[0], ce[1], ce[2]) if args.ego else (cs[0], cs[1], cs[2])
        ann_xs.append(cx); ann_ys.append(cy); ann_zs.append(cz)
        ann_labels.append(label)
        print(f"  {ann['category_name']:35s}  "
              f"sensor x={cs[0]:+7.2f} y={cs[1]:+7.2f}  |  "
              f"ego    x={ce[0]:+7.2f} y={ce[1]:+7.2f}")

    # ------------------------------------------------------------------ #
    # Cached predictions
    # ------------------------------------------------------------------ #
    pred_traces = []
    if entry is not None:
        pred_traces += _bbox_traces(
            entry.clean_predictions or [], "clean preds", "cyan", T_s2e
        )
        if entry.attacked_predictions:
            pred_traces += _bbox_traces(
                entry.attacked_predictions, "attacked preds", "orange", T_s2e
            )

    # ------------------------------------------------------------------ #
    # Front-camera image (closest CAM_FRONT sweep by timestamp)
    # ------------------------------------------------------------------ #
    if sample is not None:
        cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    else:
        # Non-keyframe: find CAM_FRONT sweep closest in timestamp
        lidar_ts = lidar_sd["timestamp"]
        cam_sds = [sd for sd in nusc.sample_data if sd["channel"] == "CAM_FRONT"]
        cam_sd = min(cam_sds, key=lambda sd: abs(sd["timestamp"] - lidar_ts))
    cam_path  = os.path.join(args.dataroot, cam_sd["filename"])

    # ------------------------------------------------------------------ #
    # Plotly figure: 3-D point cloud + annotation markers
    # ------------------------------------------------------------------ #
    import plotly.graph_objects as go
    import base64

    with open(cam_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    img_src = f"data:image/jpeg;base64,{img_b64}"

    # 3D scene occupies the left 58%; image panel fills the right 38%.
    SCENE_X_END   = 0.57
    IMAGE_X_START = 0.62

    fig = go.Figure()

    if args.patchworkpp:
        import pypatchworkpp
        params = pypatchworkpp.Parameters()
        params.sensor_height = args.sensor_height
        params.verbose = False
        with open(os.devnull, "w") as _devnull:
            _old_fd = os.dup(1)
            os.dup2(_devnull.fileno(), 1)
            try:
                ppp = pypatchworkpp.patchworkpp(params)
            finally:
                os.dup2(_old_fd, 1)
                os.close(_old_fd)
        if T_s2e is not None:
            # Apply only the rotation component of sensor→ego to level the
            # point cloud without shifting the origin away from the sensor.
            # Patchwork++ computes elevation angles relative to the origin,
            # so the origin must remain at the sensor, not move to ground level.
            xyz_level = (T_s2e[:3, :3] @ pts[:, :3].astype(np.float64).T).T.astype(np.float32)
            patchwork_input = np.hstack([xyz_level, pts[:, 3:4].astype(np.float32)])
        else:
            patchwork_input = pts[:, :4].astype(np.float32)
        ppp.estimateGround(patchwork_input)
        ground_idx    = np.asarray(ppp.getGroundIndices(),    dtype=int)
        nonground_idx = np.asarray(ppp.getNongroundIndices(), dtype=int)
        print(f"Patchwork++: {len(ground_idx):,} ground  |  {len(nonground_idx):,} non-ground")

        fig.add_trace(
            go.Scatter3d(
                x=x[ground_idx], y=y[ground_idx], z=z[ground_idx],
                mode="markers",
                marker=dict(size=1, color="lightgray"),
                name="ground (clean)",
                hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra>ground</extra>",
            )
        )
        zng = z[nonground_idx]
        fig.add_trace(
            go.Scatter3d(
                x=x[nonground_idx], y=y[nonground_idx], z=zng,
                mode="markers",
                marker=dict(size=1, color=zng, colorscale="Viridis",
                            cmin=float(zng.min()), cmax=float(zng.max()),
                            colorbar=dict(title="z (m)", thickness=12,
                                          x=SCENE_X_END - 0.01, xanchor="right",
                                          len=0.75)),
                name="non-ground (clean)",
                hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra>non-ground</extra>",
            )
        )
    else:
        # Points coloured by z
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=1, color=z, colorscale="Viridis",
                            cmin=float(z.min()), cmax=float(z.max()),
                            colorbar=dict(title="z (m)", thickness=12,
                                          x=SCENE_X_END - 0.01, xanchor="right",
                                          len=0.75)),
                name="LiDAR (clean)",
                hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
            )
        )
        if attacked_pts is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=ax, y=ay, z=az,
                    mode="markers",
                    marker=dict(size=1, color="salmon", opacity=0.6),
                    name="LiDAR (attacked)",
                    hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
                )
            )

    # GT annotation centroids
    if ann_xs:
        fig.add_trace(
            go.Scatter3d(
                x=ann_xs, y=ann_ys, z=ann_zs,
                mode="markers+text",
                marker=dict(size=6, color="red", symbol="cross"),
                text=[l.split("  ")[0] for l in ann_labels],
                textposition="top center",
                hovertext=ann_labels,
                hoverinfo="text",
                name="GT annotations",
            )
        )

    for t in pred_traces:
        fig.add_trace(t)

    # Camera image pinned to paper coordinates on the right
    fig.add_layout_image(
        dict(source=img_src,
             xref="paper", yref="paper",
             x=IMAGE_X_START, y=0.93,
             sizex=1.0 - IMAGE_X_START, sizey=0.86,
             sizing="contain",
             layer="above"),
    )

    lidar_panel_label = f"LiDAR point cloud ({'ego' if args.ego else 'sensor'} frame)"
    fig.update_layout(
        title=f"NuScenes {scene['name']} — frame {lidar_sd['token'][:8]} "
              + (f"(keyframe {sample['token'][:8]})" if sample is not None
                 else f"(nearest keyframe: {nusc.get('sample', lidar_sd['sample_token'])['token'][:8]})"),
        scene=dict(
            domain=dict(x=[0, SCENE_X_END], y=[0, 1]),
            xaxis_title="x → (forward)" if args.ego else "x →",
            yaxis_title="y → (left)"    if args.ego else "y →",
            zaxis_title="z (m)",
            aspectmode="data",
        ),
        annotations=[
            dict(text=lidar_panel_label,
                 xref="paper", yref="paper",
                 x=SCENE_X_END / 2, y=1.0,
                 xanchor="center", yanchor="bottom",
                 showarrow=False, font=dict(size=13)),
            dict(text="CAM_FRONT",
                 xref="paper", yref="paper",
                 x=IMAGE_X_START + (1.0 - IMAGE_X_START) / 2, y=1.0,
                 xanchor="center", yanchor="bottom",
                 showarrow=False, font=dict(size=13)),
        ],
        height=700,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=60, b=20, l=0, r=10),
    )

    out = "nuscenes_frame_explore.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
