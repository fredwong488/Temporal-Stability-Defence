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
import json
import pathlib
import subprocess
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
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#17becf",
        "#bcbd22", "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#2ca02c", "#1f77b4", "#ff7f0e", "#a0522d", "#6a5acd",
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
            x=xyz_past[:, 0].tolist(), y=xyz_past[:, 1].tolist(), z=xyz_past[:, 2].tolist(),
            mode="markers",
            marker=dict(size=1, color="#666666", opacity=0.4),
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
            x=npts[:, 0].tolist(), y=npts[:, 1].tolist(), z=npts[:, 2].tolist(),
            mode="markers",
            marker=dict(size=1.5, color="rgba(100,100,100,0.6)"),
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
            f"(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) m<br>"
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
            x=pts[:, 0].tolist(), y=pts[:, 1].tolist(), z=pts[:, 2].tolist(),
            mode="markers",
            marker=dict(size=size, color=color, opacity=opacity, symbol=symbol),
            name=f"C{lbl} ({status})",
            legendgroup=f"cluster_{lbl}",
            hovertemplate=hover + "<extra></extra>",
        ))

    # --- ORA injected points (if known) -------------------------------------
    if injected_xyz is not None and len(injected_xyz) > 0:
        fig.add_trace(go.Scatter3d(
            x=injected_xyz[:, 0].tolist(), y=injected_xyz[:, 1].tolist(), z=injected_xyz[:, 2].tolist(),
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
# Camera image loader
# ---------------------------------------------------------------------------

_CAM_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]
_CAM_LABELS = [
    "Front Left", "Front", "Front Right",
    "Back Left",  "Back",  "Back Right",
]


def _load_camera_images(nusc, sd_token_full: str) -> list[str | None]:
    """Return 6 base64 data-URIs for the camera views, in _CAM_ORDER. None on failure.

    Uses the most recent keyframe: walks the LiDAR sample_data prev-chain until a
    keyframe is found (avoids picking a future keyframe when sd["sample_token"]
    happens to point forward in time).
    """
    import base64
    sd = nusc.get("sample_data", sd_token_full)

    # Walk backwards through LiDAR sweeps to find the most recent keyframe.
    cur = sd
    while not cur["is_key_frame"] and cur["prev"]:
        cur = nusc.get("sample_data", cur["prev"])
    # If we exhausted prev without finding a keyframe, fall back to nearest sample.
    if not cur["is_key_frame"]:
        cur = sd
    sample = nusc.get("sample", cur["sample_token"])
    out: list[str | None] = []
    for cam in _CAM_ORDER:
        try:
            cam_sd = nusc.get("sample_data", sample["data"][cam])
            img_path = pathlib.Path(nusc.dataroot) / cam_sd["filename"]
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = img_path.suffix.lower().lstrip(".")
            mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
            out.append(f"data:image/{mime};base64,{b64}")
        except Exception:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Combined HTML writer
# ---------------------------------------------------------------------------

_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cluster Inspector</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body { margin: 0; background: #111; color: #eee; font-family: monospace;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #nav { display: flex; align-items: center; gap: 12px; padding: 6px 12px;
         background: #222; flex-shrink: 0; }
  #nav button { background: #444; color: #eee; border: 1px solid #666;
                padding: 4px 14px; cursor: pointer; font-size: 14px;
                border-radius: 3px; }
  #nav button:hover { background: #555; }
  #btn-cameras.hidden { opacity: 0.45; }
  #frame-label { font-size: 13px; flex: 1; white-space: nowrap;
                 overflow: hidden; text-overflow: ellipsis; }
  #counter { font-size: 13px; white-space: nowrap; }
  #status  { font-size: 12px; color: #aaa; white-space: nowrap; }
  #main    { display: flex; flex-direction: column; flex: 1; min-height: 0; }
  #plot    { flex: 1; min-height: 0; }
  #cameras { flex-shrink: 0; min-height: 50vh; display: grid;
             grid-template-columns: 1fr 1fr 1fr;
             grid-template-rows: 1fr 1fr;
             gap: 2px; background: #000; }
  #cameras.cam-hidden { display: none; }
  .cam-cell { position: relative; overflow: hidden; background: #222; }
  .cam-cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .cam-label { position: absolute; bottom: 2px; left: 4px; font-size: 10px;
               color: #fff; text-shadow: 0 0 3px #000; pointer-events: none; }
  .cam-missing { display: flex; align-items: center; justify-content: center;
                 width: 100%; height: 100%; font-size: 11px; color: #555; }
</style>
</head>
<body>
<div id="nav">
  <button id="btn-prev">&#9664; Prev</button>
  <button id="btn-next">Next &#9654;</button>
  <span id="frame-label"></span>
  <span id="counter"></span>
  <span id="status"></span>
  <button id="btn-cameras">&#128247; Cameras</button>
</div>
<div id="main">
  <div id="plot"></div>
  <div id="cameras"></div>
</div>
<script>
const PREFETCH = 2;
const CAM_LABELS = ['Front Left','Front','Front Right','Back Left','Back','Back Right'];
let MANIFEST = null;
let current  = 0;
const cache  = {};

async function loadManifest() {
  const r = await fetch('manifest.json');
  MANIFEST = await r.json();
  show(0);
}

async function fetchFrame(i) {
  if (cache[i]) return cache[i];
  const r = await fetch(MANIFEST[i].file);
  cache[i] = await r.json();
  return cache[i];
}

function prefetch(idx) {
  for (let d = 1; d <= PREFETCH; d++) {
    const j = (idx + d) % MANIFEST.length;
    if (!cache[j]) fetchFrame(j);
  }
}

document.getElementById('btn-cameras').addEventListener('click', () => {
  const panel = document.getElementById('cameras');
  const btn   = document.getElementById('btn-cameras');
  const hidden = panel.classList.toggle('cam-hidden');
  btn.classList.toggle('hidden', hidden);
  // Let Plotly re-fit to the newly available width
  Plotly.relayout('plot', {autosize: true});
});

function renderCameras(cameras) {
  const div = document.getElementById('cameras');
  div.innerHTML = '';
  for (let i = 0; i < 6; i++) {
    const cell = document.createElement('div');
    cell.className = 'cam-cell';
    const src = cameras && cameras[i];
    if (src) {
      const img = document.createElement('img');
      img.src = src;
      img.alt = CAM_LABELS[i];
      cell.appendChild(img);
    } else {
      const placeholder = document.createElement('div');
      placeholder.className = 'cam-missing';
      placeholder.textContent = CAM_LABELS[i];
      cell.appendChild(placeholder);
    }
    const lbl = document.createElement('span');
    lbl.className = 'cam-label';
    lbl.textContent = CAM_LABELS[i];
    cell.appendChild(lbl);
    div.appendChild(cell);
  }
}

async function show(idx) {
  if (!MANIFEST) return;
  current = ((idx % MANIFEST.length) + MANIFEST.length) % MANIFEST.length;
  document.getElementById('status').textContent = 'loading…';
  const data = await fetchFrame(current);
  const fig = data.fig || data;  // backwards-compat if fig is the whole object
  const plotDiv = document.getElementById('plot');
  const camera = plotDiv._fullLayout?.scene?.camera;
  const layout = camera
    ? Object.assign({}, fig.layout, {scene: Object.assign({}, fig.layout.scene, {camera})})
    : fig.layout;
  Plotly.react('plot', fig.data, layout, {responsive: true});
  renderCameras(data.cameras || []);
  document.getElementById('frame-label').textContent = MANIFEST[current].label;
  document.getElementById('counter').textContent =
    (current + 1) + ' / ' + MANIFEST.length;
  document.getElementById('status').textContent = '';
  prefetch(current);
}

document.getElementById('btn-prev').addEventListener('click', () => show(current - 1));
document.getElementById('btn-next').addEventListener('click', () => show(current + 1));
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  show(current - 1);
  if (e.key === 'ArrowRight') show(current + 1);
});

loadManifest();
</script>
</body>
</html>
"""


def _write_output_dir(frames_data: list[dict], out_dir: pathlib.Path) -> None:
    """Write per-frame JSON files + viewer.html into out_dir.

    Serve locally with:  python -m http.server --directory <out_dir>
    then open http://localhost:8000/viewer.html
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, fd in enumerate(frames_data):
        fname = f"{i:04d}.json"
        payload = {"fig": fd["fig_json"], "cameras": fd.get("cameras", [])}
        (frames_dir / fname).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        manifest.append({"file": f"frames/{fname}", "label": fd["label"]})

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "viewer.html").write_text(_VIEWER_HTML, encoding="utf-8")


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_metadata(
    out_dir: pathlib.Path,
    args: argparse.Namespace,
    defense_params: dict,
    attack: object,
    pipeline_kwargs: dict,
) -> None:
    meta = {
        "notes": args.notes,
        "git_commit": _git_commit(),
        "cmd_args": vars(args),
        "attack": {
            "class": type(attack).__name__,
            **{k: v for k, v in vars(attack).items() if not k.startswith("_")},
        },
        "defense_params": defense_params,
        "pipeline_params": pipeline_kwargs,
    }
    out_path = out_dir / "metadata.json"
    out_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Metadata written → {out_path}")


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

    # Frame-skipping controls (scene-granularity mode)
    p.add_argument("--skip-unattacked-frames-per-scene", type=int, default=0,
                   metavar="N",
                   help="Skip this many unattacked frames at the start of each scene (default: 0)")
    p.add_argument("--skip-attacked-frames-per-scene", type=int, default=0,
                   metavar="N",
                   help="Skip this many attacked frames at the start of each scene's attack phase (default: 0)")
    p.add_argument("--max-unattacked-frames-per-scene", type=int, default=None,
                   metavar="N",
                   help="Process at most this many unattacked frames per scene (default: all)")
    p.add_argument("--max-attacked-frames-per-scene", type=int, default=None,
                   metavar="N",
                   help="Process at most this many attacked frames per scene (default: all)")

    p.add_argument("--notes", default="",
                   help="Free-text notes to store in metadata.json alongside the run parameters")

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
    collected: list[dict] = []

    # Build frame_id (first 16 chars of sd_token) → full sd_token lookup
    nusc = getattr(dataset, "_nusc", None)
    frame_id_to_sd_token: dict[str, str] = {}
    if nusc is not None:
        for _, sd_token in dataset._entries:
            frame_id_to_sd_token[sd_token[:16]] = sd_token

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

        cameras: list[str | None] = []
        if nusc is not None:
            sd_token_full = frame_id_to_sd_token.get(frame.frame_id)
            if sd_token_full:
                cameras = _load_camera_images(nusc, sd_token_full)

        idx = len(collected) + 1
        attack_tag = "attacked" if frame.is_attacked else "clean"
        label = f"[{idx}]  {frame.frame_id[:24]}  |  {attack_tag.upper()}"
        if frame.is_attacked:
            label += "  |  defense: " + ("triggered" if result.is_attack_detected else "MISSED")
        collected.append({"fig_json": fig.to_plotly_json(), "cameras": cameras, "label": label})
        # print(f"  [{idx}] {frame.frame_id[:24]}  attack={frame.is_attacked}  defense={result.is_attack_detected}")

    defense = RadialJitterDefense(
        **defense_params,
        debug=True,
        defense_frame_hook=defense_frame_hook,
    )

    pipeline_kwargs = dict(
        precomputed_cache_path=args.precomputed_cache,
        use_cached_attacks=True,
        use_predicted_labels=True,
        attack_fraction=args.attack_fraction,
        attack_fraction_seed=args.attack_fraction_seed,
        min_unattacked_frames=args.min_unattacked_frames,
        min_attacked_frames=args.min_attacked_frames,
        max_frames=args.max_frames,
        skip_unattacked_frames_per_scene=args.skip_unattacked_frames_per_scene,
        skip_attacked_frames_per_scene=args.skip_attacked_frames_per_scene,
        max_unattacked_frames_per_scene=args.max_unattacked_frames_per_scene,
        max_attacked_frames_per_scene=args.max_attacked_frames_per_scene,
    )

    _write_metadata(out_dir, args, defense_params, attack, pipeline_kwargs)

    print("Running pipeline …")

    EvalPipeline(
        dataset=dataset,
        attack=attack,
        defense=defense,
        **pipeline_kwargs,
    ).run()

    print(f"\nWriting {len(collected)} frames → {out_dir}")
    _write_output_dir(collected, out_dir)
    print(f"Done.  Serve with:\n  python -m http.server --directory {out_dir}\nthen open http://localhost:8000/viewer.html")


if __name__ == "__main__":
    main()
