"""
tools/inspect_clusters_3d.py
-----------------------------
Run radial_jitter on a NuScenes sequence and save one interactive Plotly
HTML file per frame, showing the full DBSCAN-clustered point cloud coloured by
cluster ID with flagged clusters highlighted.

Designed to be run on a remote server; copy the output HTML files locally and
open in any browser — no dependencies required on the viewing machine.

Typical usage (mirrors the sweep command):

    pixi run python tools/inspect_clusters_3d.py \\
        --precomputed-cache precomputed/nuscenes-pointpillars-ora-withnoise-b200-0.5/defense_sweep_shared.pkl \\
        --attack-fraction 0.5 \\
        --output-dir /tmp/clusters_3d \\
        --max-frames 40 \\
        --defense-params dbscan_eps=0.5 temporal_window=8 centroid_method=first_diff use_point=False
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from eval_pipeline.datasets.nuscenes import NuScenesDataset
from eval_pipeline.attacks.ora import ORAAttack
from eval_pipeline.defenses.radial_jitter import RadialJitterDefense
from eval_pipeline.pipeline import EvalPipeline
from eval_pipeline.types import DetectionResult, Frame
from eval_pipeline.utils.spoofing_noise import SpoofingNoiseModel


def _parse_kv_params(pairs: list[str]) -> dict:
    """Parse KEY=VALUE strings into a dict, auto-casting values to bool/int/float/str."""
    out: dict = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid parameter '{item}': expected KEY=VALUE format"
            )
        key, _, raw = item.partition("=")
        if raw.lower() == "true":
            out[key] = True
        elif raw.lower() == "false":
            out[key] = False
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out


# ---------------------------------------------------------------------------
# Defaults matching the sweep script
# ---------------------------------------------------------------------------
_DATASETS_BASE = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets"
_DEFAULT_NUSCENES_ROOT    = f"{_DATASETS_BASE}/nuscenes-v1.0-mini"
_DEFAULT_NUSCENES_VERSION = "v1.0-mini"
_DEFAULT_NUSCENES_SPLIT   = "mini_val"
NUSCENES_DEFAULT_CLASSES = ["car", "pedestrian", "bicycle"]


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _cluster_colors(n: int) -> list[str]:
    """Distinct colours for up to n clusters (cycles if n > palette)."""
    palette = [
        "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        "#17becf", "#bcbd22", "#7f7f7f", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#2ca02c", "#1f77b4", "#ff7f0e",
    ]
    return [palette[i % len(palette)] for i in range(n)]


def _make_frame_figure(
    cur_xyz_filt: np.ndarray,
    labels_cur: np.ndarray,
    cluster_details: list[dict],
    past_xyz_list: list[np.ndarray],
    injected_xyz: np.ndarray | None,
    frame_id: str,
    is_attacked: bool,
    is_defense_triggered: bool,
    defense_params: dict,
):
    """Build a Plotly Figure with one trace per cluster."""
    import plotly.graph_objects as go

    unique_labels = sorted(lbl for lbl in set(labels_cur) if lbl != -1)
    colors = _cluster_colors(len(unique_labels))
    label_to_color = {lbl: col for lbl, col in zip(unique_labels, colors)}

    # Map cluster index → detail dict (details are in order of unique_labels from
    # the defense, which iterates set(labels_cur)-{-1}).  We sort both the same
    # way so indices align.
    detail_map: dict[int, dict] = {}
    for i, lbl in enumerate(unique_labels):
        if i < len(cluster_details):
            detail_map[lbl] = cluster_details[i]

    fig = go.Figure()

    # --- past sweeps (light gray, small) ------------------------------------
    for t, xyz_past in enumerate(past_xyz_list):
        if len(xyz_past) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=xyz_past[:, 0], y=xyz_past[:, 1], z=xyz_past[:, 2],
            mode="markers",
            marker=dict(size=1, color="lightgray", opacity=0.15),
            name=f"Past t-{len(past_xyz_list)-t}",
            legendgroup="past",
            showlegend=(t == 0),
            hoverinfo="skip",
        ))

    # --- noise points -------------------------------------------------------
    noise_mask = labels_cur == -1
    if noise_mask.any():
        npts = cur_xyz_filt[noise_mask]
        fig.add_trace(go.Scatter3d(
            x=npts[:, 0], y=npts[:, 1], z=npts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="rgba(150,150,150,0.25)"),
            name="Noise (unclustered)",
            legendgroup="noise",
        ))

    # --- cluster points ------------------------------------------------------
    for lbl in unique_labels:
        mask = labels_cur == lbl
        pts  = cur_xyz_filt[mask]
        det  = detail_map.get(lbl, {})
        flagged   = det.get("flagged", False)
        skipped   = det.get("skipped")
        sigma_c   = det.get("sigma_centroid")
        sigma_p   = det.get("sigma_point")
        n_frames  = det.get("n_frames_associated", 0)
        n_pts_cur = det.get("n_points_cur", len(pts))

        status = "flagged" if flagged else ("skipped: " + skipped if skipped else "ok")
        hover = (
            f"Cluster {lbl}<br>"
            f"Points: {n_pts_cur}<br>"
            f"Frames assoc.: {n_frames}<br>"
            f"σ_centroid: {f'{sigma_c:.4f}' if sigma_c is not None else 'n/a'}<br>"
            f"σ_point: {f'{sigma_p:.4f}' if sigma_p is not None else 'n/a'}<br>"
            f"Status: {status}"
        )

        color   = label_to_color[lbl]
        size    = 2.5
        symbol  = "circle"
        opacity = 0.8

        if skipped:
            opacity = 0.35
            size    = 1.5
        elif flagged:
            color   = "red"
            size    = 3.5
            symbol  = "diamond"

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=opacity, symbol=symbol),
            name=f"C{lbl} ({status})",
            legendgroup=f"cluster_{lbl}",
            hovertemplate=hover + "<extra></extra>",
        ))

    # --- ORA injected points (if known) -------------------------------------
    if injected_xyz is not None and len(injected_xyz) > 0:
        fig.add_trace(go.Scatter3d(
            x=injected_xyz[:, 0], y=injected_xyz[:, 1], z=injected_xyz[:, 2],
            mode="markers",
            marker=dict(size=4, color="magenta", symbol="x", opacity=0.9),
            name="ORA injected",
            legendgroup="ora",
        ))

    # --- cluster centroids --------------------------------------------------
    for lbl in unique_labels:
        det = detail_map.get(lbl, {})
        if "centroid" not in det:
            continue
        c = det["centroid"]
        flagged = det.get("flagged", False)
        skipped = det.get("skipped")
        marker_color = "red" if flagged else ("gray" if skipped else label_to_color[lbl])
        fig.add_trace(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[c[2]],
            mode="markers+text",
            marker=dict(size=7, color=marker_color, symbol="cross", opacity=1.0,
                        line=dict(color="black", width=1)),
            text=[f"C{lbl}"],
            textposition="top center",
            name=f"Centroid C{lbl}",
            legendgroup=f"cluster_{lbl}",
            showlegend=False,
            hovertemplate=f"Centroid C{lbl}<extra></extra>",
        ))

    attack_str = "YES (defense triggered)" if is_attacked and is_defense_triggered \
        else ("YES (missed)" if is_attacked else "no")
    title = (
        f"Frame {frame_id} | Attack: {attack_str} | {len(unique_labels)} clusters"
    )
    params_text = "<br>".join(f"{k}: {v}" for k, v in sorted(defense_params.items()))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m, forward)",
            yaxis_title="Y (m, left)",
            zaxis_title="Z (m, up)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
        annotations=[dict(
            text=params_text,
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            xanchor="left", yanchor="bottom",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10, family="monospace"),
            align="left",
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise radial_jitter DBSCAN clusters as interactive 3D HTML files."
    )

    # Dataset
    p.add_argument("--nuscenes-root",    default=_DEFAULT_NUSCENES_ROOT)
    p.add_argument("--nuscenes-version", default=_DEFAULT_NUSCENES_VERSION)
    p.add_argument("--nuscenes-split",   default=_DEFAULT_NUSCENES_SPLIT)
    p.add_argument("--scene-names",      nargs="+", default=None,
                   help="Restrict to specific scene names (default: all in split)")
    
    # Pipeline
    p.add_argument("--min-unattacked-frames", type=int, default=6,
                        metavar="N",
                        help="Minimum frames left unattacked at the start of each attacked scene "
                             "(NuScenes / scene-granularity datasets only). "
                             "Actual prefix is randomised in [N, scene_length - min-attacked-frames].")
    p.add_argument("--min-attacked-frames", type=int, default=6,
                        metavar="N",
                        help="Minimum frames that must be attacked in a chosen scene. "
                             "Scenes too short to satisfy both minima revert to unattacked.")

    # Precomputed cache (for attack metadata + clean predictions)
    p.add_argument("--precomputed-cache", default=None,
                   help="Path to a precomputed *.pkl cache (e.g. defense_sweep_shared.pkl)")
    p.add_argument("--attack-fraction", type=float, default=0.5,
                   help="Fraction of scenes to attack (must match the original run)")
    p.add_argument("--attack-fraction-seed", type=int, default=0)

    # Defense params
    p.add_argument(
        "--defense-params", nargs="*", default=[], metavar="KEY=VALUE",
        help="RadialJitterDefense constructor kwargs as KEY=VALUE pairs "
             "(e.g. --defense-params dbscan_eps=0.5 temporal_window=8). "
             "Values are auto-cast to bool, int, float, or str.",
    )

    # Output
    p.add_argument("--output-dir",  default="cluster_viz_3d",
                   help="Directory to write HTML files into")
    p.add_argument("--max-frames",  type=int, default=None,
                   help="Stop after this many frames (across all scenes)")
    p.add_argument("--attacked-only", action="store_true",
                   help="Only save HTML for frames where an attack is active")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading NuScenes {args.nuscenes_split} from {args.nuscenes_root} …")
    dataset_kwargs: dict = dict(
        root=args.nuscenes_root,
        version=args.nuscenes_version,
        split=args.nuscenes_split,
    )
    if args.scene_names:
        dataset_kwargs["scene_names"] = args.scene_names
    dataset = NuScenesDataset(**dataset_kwargs)

    attack = ORAAttack(budget=200, target_types=NUSCENES_DEFAULT_CLASSES, noise_model=SpoofingNoiseModel.from_preset("worst_case", seed=args.attack_fraction_seed), debug=True)

    defense_params = _parse_kv_params(args.defense_params)
    n_saved = [0]

    def defense_frame_hook(
        frame: Frame,
        result: DetectionResult,
        past_xyz_list: list[np.ndarray],
    ) -> None:
        if args.attacked_only and not frame.is_attacked:
            return

        cur_xyz_filt = result.metadata["xyz_filt"]
        labels_cur   = result.metadata["labels_cur"]
        cluster_details = result.metadata.get("cluster_details", [])

        injected_xyz: np.ndarray | None = None
        meta = frame.attack_metadata
        if meta:
            parts = [
                obj["reinjected_xyz"]
                for obj in meta.get("removed_per_obj", [])
                if "reinjected_xyz" in obj
            ]
            if parts:
                injected_xyz = np.array([pt for xyz in parts for pt in xyz], dtype=np.float32)

        fig = _make_frame_figure(
            cur_xyz_filt=cur_xyz_filt,
            labels_cur=labels_cur,
            cluster_details=cluster_details,
            past_xyz_list=past_xyz_list,
            injected_xyz=injected_xyz,
            frame_id=frame.frame_id,
            is_attacked=frame.is_attacked,
            is_defense_triggered=result.is_attack_detected,
            defense_params=defense_params,
        )

        n_saved[0] += 1
        attack_tag = "attacked" if frame.is_attacked else "clean"
        html_name = f"{n_saved[0]:04d}_{frame.frame_id[:16]}_{attack_tag}.html"
        fig.write_html(str(out_dir / html_name), include_plotlyjs="cdn")
        print(f"  [{n_saved[0]}] {html_name}  attack={frame.is_attacked}  defense={result.is_attack_detected}")

    defense = RadialJitterDefense(
        **defense_params,
        debug=True,
        defense_frame_hook=defense_frame_hook,
    )

    print(f"Writing HTML files to {out_dir}/")

    EvalPipeline(
        dataset=dataset,
        attack=attack,
        defense=defense,
        precomputed_cache_path=args.precomputed_cache,
        use_cached_attacks=True,
        attack_fraction=args.attack_fraction,
        attack_fraction_seed=args.attack_fraction_seed,
        min_unattacked_frames=args.min_unattacked_frames,
        min_attacked_frames=args.min_attacked_frames,
        max_frames=args.max_frames,
    ).run()

    print(f"\nDone. {n_saved[0]} HTML files written to {out_dir}/")


if __name__ == "__main__":
    main()
