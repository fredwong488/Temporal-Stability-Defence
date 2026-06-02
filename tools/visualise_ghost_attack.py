"""
tools/visualise_ghost_attack.py
--------------------------------
Visualise the difference between clean LIDAR_TOP sweep point clouds and
ghost-attacked counterparts from sweeps/LIDAR_TOP_GHOST_ATTACK/{CAR,CYL,PED}.

Ghost-injected points (present in the attack cloud but absent from the clean
cloud) are highlighted in red.  Clean points are shown in gray/viridis.

Output: interactive HTML viewer using the same setup as inspect_clusters_3d.py.

Usage:
    pixi run python tools/visualise_ghost_attack.py \\
        --attack-type car \\
        --frames-per-scene 20 \\
        --output-dir /tmp/ghost_viz

    python -m http.server --directory /tmp/ghost_viz
    # open http://localhost:8000/viewer.html

    # Stats mode (no output saved):
    pixi run python tools/visualise_ghost_attack.py --attack-type car --stats
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_NUSCENES_ROOT = str(_ROOT / "data/datasets/nuscenes-v1.0-mini")
_GOA_TRACES_DIR = _ROOT / "eval_pipeline/attacks/ghost_object/traces"

_ATTACK_SUBDIR = {
    "car": "LIDAR_TOP_ATTACK_CAR",
    "cyl": "LIDAR_TOP_ATTACK_CYL",
    "ped": "LIDAR_TOP_ATTACK_PED",
}


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------

def _load_bin(path: pathlib.Path) -> np.ndarray:
    """Load a NuScenes .pcd.bin as (N, 4) float32 [x, y, z, intensity]."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]


def _find_diff_masks(
    clean: np.ndarray, attacked: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ghost_mask, removed_mask).

    ghost_mask   — bool mask over *attacked* rows absent from *clean* (injected).
    removed_mask — bool mask over *clean* rows absent from *attacked* (deleted).

    Coordinates are rounded to 3 decimal places (sub-mm) before comparison to
    guard against any float32 repr differences.
    """
    scale = 1_000.0
    clean_int = np.round(clean[:, :3].astype(np.float64) * scale).astype(np.int64)
    attacked_int = np.round(attacked[:, :3].astype(np.float64) * scale).astype(np.int64)

    clean_set: set[tuple] = set(map(tuple, clean_int.tolist()))
    attacked_set: set[tuple] = set(map(tuple, attacked_int.tolist()))

    ghost = np.array([tuple(r) not in clean_set for r in attacked_int.tolist()], dtype=bool)
    removed = np.array([tuple(r) not in attacked_set for r in clean_int.tolist()], dtype=bool)
    return ghost, removed


# ---------------------------------------------------------------------------
# Scene/frame discovery
# ---------------------------------------------------------------------------

def _discover_attacked_scenes(
    attack_dir: pathlib.Path,
    sweeps_dir: pathlib.Path,
    samples_dir: pathlib.Path,
) -> dict[str, list[tuple[pathlib.Path | None, pathlib.Path]]]:
    """Return {scene_prefix: sorted list of (clean_path_or_None, attack_path)}.

    Attacked frames live in attack_dir.  Clean counterparts are looked up first
    in sweeps_dir (LIDAR_TOP sweeps), then in samples_dir (LIDAR_TOP samples),
    to handle the case where some keyframes are stored under samples/.
    """
    scenes: dict[str, list[tuple[pathlib.Path | None, pathlib.Path]]] = {}
    for attack_path in sorted(attack_dir.glob("*.pcd.bin")):
        prefix = attack_path.name.split("__LIDAR_TOP__")[0]
        clean_path: pathlib.Path | None = sweeps_dir / attack_path.name
        if not clean_path.exists():
            candidate = samples_dir / attack_path.name
            clean_path = candidate if candidate.exists() else None
        scenes.setdefault(prefix, []).append((clean_path, attack_path))
    for v in scenes.values():
        v.sort(key=lambda t: t[1].name)
    return scenes


def _evenly_spaced(items: list, n: int | None) -> list:
    """Pick n evenly-spaced elements from items (inclusive of first and last).

    If n is None, return all items.
    """
    if n is None or len(items) <= n:
        return list(items)
    indices = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
    return [items[i] for i in indices]


# ---------------------------------------------------------------------------
# GhostObjectAttack runner
# ---------------------------------------------------------------------------

def _load_goa_attack(attack_type: str):
    """Instantiate GhostObjectAttack for *attack_type*, or raise FileNotFoundError."""
    from eval_pipeline.attacks.ghost_object.ghost_object import GhostObjectAttack
    npy_path = _GOA_TRACES_DIR / f"ghost_cloud_{attack_type}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"GOA ghost cloud not found: {npy_path}\n"
            "Generate it with: pixi run python tools/visualise_ghost_attack.py "
            f"--attack-type {attack_type} --ghost-cloud-output eval_pipeline/attacks/ghost_object/traces"
        )
    return GhostObjectAttack(ghost_cloud_path=npy_path)


def _apply_goa(attack, clean: np.ndarray, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the attack on *clean* and return (goa_injected, goa_removed) point arrays."""
    from eval_pipeline.types import Frame
    frame = Frame(
        frame_id=frame_name,
        sequence_id="",
        timestamp=0.0,
        lidar=clean,
        image=None,
        labels=[],
    )
    attacked_frame = attack.apply(frame)
    goa_injected, goa_removed = _find_diff_masks(clean, attacked_frame.lidar)
    return attacked_frame.lidar[goa_injected], clean[goa_removed]


# ---------------------------------------------------------------------------
# Plotly figure builder
# ---------------------------------------------------------------------------

def _make_diff_figure(
    clean: np.ndarray,
    ghost_pts: np.ndarray,
    removed_pts: np.ndarray,
    scene: str,
    filename: str,
    frame_idx: int,
    total_frames: int,
    attack_type: str,
    goa_injected: np.ndarray | None = None,
    goa_removed: np.ndarray | None = None,
) -> dict:
    """Build a Plotly figure dict with clean + ghost + removed traces."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Clean points — coloured by intensity (viridis)
    fig.add_trace(go.Scatter3d(
        x=clean[:, 0].tolist(),
        y=clean[:, 1].tolist(),
        z=clean[:, 2].tolist(),
        mode="markers",
        marker=dict(
            size=1.5,
            color=clean[:, 3].tolist(),
            colorscale="Viridis",
            opacity=0.5,
            colorbar=dict(title="Intensity", thickness=12, len=0.5),
        ),
        name=f"Clean ({len(clean):,} pts)",
        hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>clean</extra>",
    ))

    # Ghost-injected points — red, larger
    if len(ghost_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=ghost_pts[:, 0].tolist(),
            y=ghost_pts[:, 1].tolist(),
            z=ghost_pts[:, 2].tolist(),
            mode="markers",
            marker=dict(size=3.5, color="red", opacity=0.9, symbol="diamond"),
            name=f"Ghost injected ({len(ghost_pts):,} pts)",
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>ghost</extra>",
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode="markers",
            marker=dict(size=3, color="red"),
            name="Ghost injected (0 pts)",
        ))

    # Removed points — orange, larger
    if len(removed_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=removed_pts[:, 0].tolist(),
            y=removed_pts[:, 1].tolist(),
            z=removed_pts[:, 2].tolist(),
            mode="markers",
            marker=dict(size=3.5, color="orange", opacity=0.9, symbol="diamond"),
            name=f"Removed ({len(removed_pts):,} pts)",
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>removed</extra>",
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode="markers",
            marker=dict(size=3, color="orange"),
            name="Removed (0 pts)",
        ))

    # GhostObjectAttack injected — cyan
    if goa_injected is not None:
        n = len(goa_injected)
        fig.add_trace(go.Scatter3d(
            x=goa_injected[:, 0].tolist() if n else [],
            y=goa_injected[:, 1].tolist() if n else [],
            z=goa_injected[:, 2].tolist() if n else [],
            mode="markers",
            marker=dict(size=3.5, color="cyan", opacity=0.9, symbol="circle"),
            name=f"GOA injected ({n:,} pts)",
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>GOA injected</extra>",
        ))

    # GhostObjectAttack removed — magenta
    if goa_removed is not None:
        n = len(goa_removed)
        fig.add_trace(go.Scatter3d(
            x=goa_removed[:, 0].tolist() if n else [],
            y=goa_removed[:, 1].tolist() if n else [],
            z=goa_removed[:, 2].tolist() if n else [],
            mode="markers",
            marker=dict(size=3.5, color="magenta", opacity=0.9, symbol="circle"),
            name=f"GOA removed ({n:,} pts)",
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>GOA removed</extra>",
        ))

    fig.update_layout(
        title=(
            f"Ghost attack — {attack_type.upper()} | "
            f"Scene: {scene} | Frame {frame_idx + 1}/{total_frames} | "
            f"{filename}"
        ),
        scene=dict(
            xaxis_title="X (m, forward)",
            yaxis_title="Y (m, left)",
            zaxis_title="Z (m, up)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig.to_plotly_json()


# ---------------------------------------------------------------------------
# HTML viewer — identical to inspect_clusters_3d.py
# ---------------------------------------------------------------------------

_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ghost Attack Viewer</title>
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
  #frame-label { font-size: 13px; flex: 1; white-space: nowrap;
                 overflow: hidden; text-overflow: ellipsis; }
  #counter { font-size: 13px; white-space: nowrap; }
  #status  { font-size: 12px; color: #aaa; white-space: nowrap; }
  #main    { display: flex; flex-direction: column; flex: 1; min-height: 0; }
  #plot    { flex: 1; min-height: 0; }
</style>
</head>
<body>
<div id="nav">
  <button id="btn-prev">&#9664; Prev</button>
  <button id="btn-next">Next &#9654;</button>
  <span id="frame-label"></span>
  <span id="counter"></span>
  <span id="status"></span>
</div>
<div id="main">
  <div id="plot"></div>
</div>
<script>
const PREFETCH = 2;
let MANIFEST = null;
let current  = 0;
const cache  = {};

const BUST = '?t=' + Date.now();

async function loadManifest() {
  const r = await fetch('manifest.json' + BUST);
  MANIFEST = await r.json();
  show(0);
}

async function fetchFrame(i) {
  if (cache[i]) return cache[i];
  const r = await fetch(MANIFEST[i].file + BUST);
  cache[i] = await r.json();
  return cache[i];
}

function prefetch(idx) {
  for (let d = 1; d <= PREFETCH; d++) {
    const j = (idx + d) % MANIFEST.length;
    if (!cache[j]) fetchFrame(j);
  }
}

async function show(idx) {
  if (!MANIFEST) return;
  current = ((idx % MANIFEST.length) + MANIFEST.length) % MANIFEST.length;
  document.getElementById('status').textContent = 'loading…';
  const data = await fetchFrame(current);
  const fig = data.fig || data;
  const plotDiv = document.getElementById('plot');
  const camera = plotDiv._fullLayout?.scene?.camera;
  const layout = camera
    ? Object.assign({}, fig.layout, {scene: Object.assign({}, fig.layout.scene, {camera})})
    : fig.layout;
  Plotly.react('plot', fig.data, layout, {responsive: true});
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


def _write_output(frames_data: list[dict], out_dir: pathlib.Path) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, fd in enumerate(frames_data):
        fname = f"{i:04d}.json"
        payload = {"fig": fd["fig_json"]}
        (frames_dir / fname).write_text(json.dumps(payload), encoding="utf-8")
        manifest.append({"file": f"frames/{fname}", "label": fd["label"]})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "viewer.html").write_text(_VIEWER_HTML, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_ghost_cloud(
    ghost_candidates: list[np.ndarray],
    output_path: pathlib.Path,
) -> None:
    """Save the first frame's ghost points as .npy.

    The ghost object is identical in every frame, so one frame is sufficient.
    The saved array is (N, 4) float32 [x, y, z, intensity] in the original
    sensor frame coordinates.
    """
    if not ghost_candidates:
        print("  [warn] no ghost points found across any frame — skipping cloud save")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), ghost_candidates[0].astype(np.float32))
    print(f"  Ghost cloud saved → {output_path}  ({len(ghost_candidates[0]):,} pts)")



def _print_count_centroid_stats(label: str, counts: list[int], centroids: list[tuple]) -> None:
    count_freq: dict[int, int] = collections.Counter(counts)
    print(f"{label} — point counts per frame:")
    for n_pts, n_frames in sorted(count_freq.items(), key=lambda x: -x[1]):
        print(f"  {n_pts} points × {n_frames} frames")

    print()
    centroid_freq: dict[tuple, int] = collections.Counter(centroids)
    print(f"{label} — centroid positions per frame:")
    for centroid, n_frames in sorted(centroid_freq.items(), key=lambda x: -x[1]):
        x, y, z = centroid
        print(f"  ({x:.3f}, {y:.3f}, {z:.3f}) × {n_frames} frames")


def _print_stats(
    ghost_point_counts: list[int],
    ghost_centroids: list[tuple[float, float, float]],
    removed_point_counts: list[int],
    removed_centroids: list[tuple[float, float, float]],
) -> None:
    _print_count_centroid_stats("Injected (ghost)", ghost_point_counts, ghost_centroids)
    print()
    _print_count_centroid_stats("Removed (clean − dirty)", removed_point_counts, removed_centroids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise clean vs ghost-attacked LiDAR point clouds interactively."
    )
    p.add_argument("--nuscenes-root", default=_DEFAULT_NUSCENES_ROOT,
                   help="Path to the NuScenes dataset root")
    p.add_argument("--attack-type", choices=["car", "cyl", "ped"], default="car",
                   help="Ghost attack variant to compare against (default: car)")
    p.add_argument("--frames-per-scene", type=int, default=None,
                   help="Number of evenly-spaced frames to sample per scene (default: all frames)")
    p.add_argument("--scene", default=None,
                   help="Restrict to a single scene prefix (e.g. n008-2018-08-01-15-16-36-0400)")
    p.add_argument("--output-dir", default="ghost_viz",
                   help="Directory to write the viewer into (default: ghost_viz); "
                        "pass '' to skip visualisation entirely")
    p.add_argument("--ghost-cloud-output", default=None,
                   help="Path to save the extracted ghost point cloud as .npy "
                        "(default: <output-dir>/ghost_cloud_<attack-type>.npy)")
    p.add_argument("--stats", action="store_true",
                   help="Print ghost-point count and centroid statistics instead of generating a viewer")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = pathlib.Path(args.nuscenes_root)
    sweeps_lidar_dir = root / "sweeps" / "LIDAR_TOP"
    samples_lidar_dir = root / "samples" / "LIDAR_TOP"
    attack_dir = root / "sweeps" / "LIDAR_TOP_GHOST_ATTACK" / _ATTACK_SUBDIR[args.attack_type]

    if not sweeps_lidar_dir.is_dir():
        sys.exit(f"LIDAR_TOP sweeps directory not found: {sweeps_lidar_dir}")
    if not attack_dir.is_dir():
        sys.exit(f"Attack directory not found: {attack_dir}")

    save_viz = bool(args.output_dir) and not args.stats
    save_clouds = args.ghost_cloud_output is not None and not args.stats

    out_dir: pathlib.Path | None = None
    if save_viz:
        out_dir = pathlib.Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    ghost_cloud_out: pathlib.Path | None = None
    if save_clouds:
        ghost_cloud_out = pathlib.Path(args.ghost_cloud_output) / f"ghost_cloud_{args.attack_type}.npy"

    scenes = _discover_attacked_scenes(attack_dir, sweeps_lidar_dir, samples_lidar_dir)
    if args.scene:
        if args.scene not in scenes:
            sys.exit(f"Scene '{args.scene}' not found. Available: {list(scenes)}")
        scenes = {args.scene: scenes[args.scene]}

    goa_attack = None
    if save_viz and not args.stats:
        goa_attack = _load_goa_attack(args.attack_type)

    frames_label = str(args.frames_per_scene) if args.frames_per_scene is not None else "all"
    print(f"Attack type : {args.attack_type.upper()}  ({attack_dir.name})")
    print(f"Scenes      : {len(scenes)}")
    print(f"Frames/scene: {frames_label}")
    if args.stats:
        print("Mode        : stats (no output written)")
    elif not save_viz and not save_clouds:
        print("Mode        : no output (pass --output-dir or --ghost-cloud-output to save)")
    print()

    collected: list[dict] = []
    ghost_candidates: list[np.ndarray] = []
    ghost_point_counts: list[int] = []
    ghost_centroids: list[tuple[float, float, float]] = []
    removed_point_counts: list[int] = []
    removed_centroids: list[tuple[float, float, float]] = []

    for scene_prefix, all_frame_pairs in scenes.items():
        sampled = _evenly_spaced(all_frame_pairs, args.frames_per_scene)
        print(f"  {scene_prefix}  ({len(all_frame_pairs)} total → {len(sampled)} sampled)")

        for fi, (clean_path, attack_path) in enumerate(sampled):
            if clean_path is None:
                print(f"    [skip] no clean file found for {attack_path.name}")
                continue

            clean = _load_bin(clean_path)
            attacked = _load_bin(attack_path)

            ghost_mask, removed_mask = _find_diff_masks(clean, attacked)
            ghost_pts = attacked[ghost_mask]
            removed_pts = clean[removed_mask]

            ghost_point_counts.append(len(ghost_pts))
            if len(ghost_pts) > 0:
                cx, cy, cz = ghost_pts[:, :3].mean(axis=0).tolist()
                ghost_centroids.append((round(cx, 3), round(cy, 3), round(cz, 3)))
            else:
                ghost_centroids.append((float("nan"), float("nan"), float("nan")))

            removed_point_counts.append(len(removed_pts))
            if len(removed_pts) > 0:
                cx, cy, cz = removed_pts[:, :3].mean(axis=0).tolist()
                removed_centroids.append((round(cx, 3), round(cy, 3), round(cz, 3)))
            else:
                removed_centroids.append((float("nan"), float("nan"), float("nan")))

            if save_clouds and len(ghost_pts) > 0 and not ghost_candidates:
                ghost_candidates.append(ghost_pts)

            if save_viz:
                goa_injected, goa_removed = None, None
                if goa_attack is not None:
                    goa_injected, goa_removed = _apply_goa(goa_attack, clean, clean_path.name)
                fig_json = _make_diff_figure(
                    clean=clean,
                    ghost_pts=ghost_pts,
                    removed_pts=removed_pts,
                    scene=scene_prefix,
                    filename=clean_path.name,
                    frame_idx=fi,
                    total_frames=len(sampled),
                    attack_type=args.attack_type,
                    goa_injected=goa_injected,
                    goa_removed=goa_removed,
                )
                label = (
                    f"{scene_prefix} | frame {fi + 1}/{len(sampled)} | "
                    f"clean={len(clean):,}  ghost={len(ghost_pts):,}"
                )
                collected.append({"fig_json": fig_json, "label": label})

            print(f"    [{fi + 1:2d}/{len(sampled)}] clean={len(clean):,}  ghost={len(ghost_pts):,}  removed={len(removed_pts):,}")

    print()
    if args.stats:
        _print_stats(ghost_point_counts, ghost_centroids, removed_point_counts, removed_centroids)
    else:
        if save_viz:
            print(f"Writing {len(collected)} frames → {out_dir}")
            _write_output(collected, out_dir)
            print(f"Viewer ready.  Serve with:\n  python -m http.server --directory {out_dir}\nthen open http://localhost:8000/viewer.html")
        if save_clouds:
            _save_ghost_cloud(ghost_candidates, ghost_cloud_out)


if __name__ == "__main__":
    main()
