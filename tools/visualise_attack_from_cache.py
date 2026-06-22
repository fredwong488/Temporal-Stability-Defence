"""
tools/visualise_attack_from_cache.py
-------------------------------------
Visualise what an attack did to LiDAR frames of a *finished* experiment run,
using only:

  - the precomputed shelve cache the run consumed
    (precomputed/<name>/single_run.{dat,dir,bak}), which stores per-frame
    ``FrameCacheEntry`` objects (attacked_lidar, clean/attacked predictions,
    attack_metadata), and
  - the run's ``single_run_frames.jsonl`` (per-frame defense outcomes + scene
    position).

Clean point clouds are read from the NuScenes dataset and diffed against the
cached ``attacked_lidar`` exactly like ``tools/visualise_ghost_attack.py``
(clean = viridis, injected = red, removed = orange).  The detector's clean and
attacked prediction boxes from the cache are overlaid (cyan / orange).

Like ``tools/plot_roc.py`` the tool lists the available experiments with their
run notes, then automatically resolves which precomputed cache the chosen run
used (printing the cache path), and lists the scenes in the run together with
their recomputed per-scene defense precision / recall / F1 (frames.jsonl has no
scene-level metrics).  You then pick one or more scenes to render.

Usage:
    pixi run python tools/visualise_attack_from_cache.py
    pixi run python tools/visualise_attack_from_cache.py \\
        --run-dir results/2026-06-11-00-36-35 \\
        --precomputed-root /vol/bitbucket/.../precomputed \\
        --scene scene-0077,scene-0078 --frames-per-scene 20

    python -m http.server --directory results/<run>/cache_attack_viz
    # open http://localhost:8000/viewer.html
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shelve
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from eval_pipeline.metrics.common import _defense_metrics_from_pairs  # noqa: E402

# Reuse the diff/figure machinery and bbox traces from the existing tools.
# The viewer HTML + writer are defined locally (below) so this tool can persist
# legend visibility across frames without changing visualise_ghost_attack.py.
from tools.visualise_ghost_attack import (  # noqa: E402
    _evenly_spaced,
    _find_diff_masks,
    _make_diff_figure,
)
from tools.explore_nuscenes_frame import (  # noqa: E402
    _bbox_traces,
    global_to_sensor,
)

RESULTS_DIR = _ROOT / "results"
_CACHE_STEM = "single_run"  # shelve files are <cache_dir>/single_run.{dat,dir,bak}


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _load_metadata(run_dir: pathlib.Path) -> dict | None:
    p = run_dir / "run_metadata.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _frames_jsonl(run_dir: pathlib.Path) -> pathlib.Path | None:
    matches = sorted(run_dir.glob("*_frames.jsonl"))
    return matches[0] if matches else None


def _resolve_cache_name(meta: dict) -> str:
    """Derive the actual precomputed cache directory name from run metadata.

    The recorded ``precomputed_cache_dir`` basename is not always the real
    cache: ghost runs record the generic ``...-ghost-1.0`` while the on-disk
    cache encodes the variant (car/ped/cyl), taken from the ghost cloud file.
    """
    cache_dir = meta.get("precomputed_cache_dir") or ""
    base = pathlib.PurePosixPath(cache_dir).name

    if meta.get("attack_type") == "ghost":
        ghost_file = (meta.get("attack_params") or {}).get("ghost_cloud_file", "")
        # ghost_cloud_car.npy -> car
        stem = pathlib.PurePosixPath(ghost_file).stem
        variant = stem.replace("ghost_cloud_", "") if stem else ""
        if variant and "-ghost-" in base and f"-ghost-{variant}-" not in base:
            base = base.replace("-ghost-", f"-ghost-{variant}-")
    return base


def _resolve_cache_stem(
    meta: dict, precomputed_root: pathlib.Path | None
) -> pathlib.Path:
    """Return the shelve stem path (<root>/<cache_name>/single_run) or exit."""
    cache_dir = meta.get("precomputed_cache_dir") or ""
    if precomputed_root is None:
        if not cache_dir:
            sys.exit("No precomputed_cache_dir in metadata and no --precomputed-root given.")
        precomputed_root = pathlib.Path(cache_dir).parent

    cache_name = _resolve_cache_name(meta)
    cache_dir_path = precomputed_root / cache_name
    stem = cache_dir_path / _CACHE_STEM

    if not (cache_dir_path.is_dir() and any(cache_dir_path.glob(f"{_CACHE_STEM}.*"))):
        available = (
            sorted(p.name for p in precomputed_root.iterdir() if p.is_dir())
            if precomputed_root.is_dir()
            else []
        )
        sys.exit(
            f"Precomputed cache '{cache_name}' not found under {precomputed_root}.\n"
            f"Available: {available}\n"
            "Pass --precomputed-root to point at the directory holding the caches."
        )
    return stem


# ---------------------------------------------------------------------------
# Run picker (mirrors tools/plot_roc.py)
# ---------------------------------------------------------------------------

def _list_runs(results_dir: pathlib.Path) -> list[pathlib.Path]:
    """Result dirs with a frames.jsonl AND a precomputed_cache_dir in metadata."""
    dirs = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        if _frames_jsonl(d) is None:
            continue
        meta = _load_metadata(d)
        if meta and meta.get("precomputed_cache_dir"):
            dirs.append(d)
    return dirs


def _pick_run_dir(results_dir: pathlib.Path) -> pathlib.Path:
    dirs = _list_runs(results_dir)
    if not dirs:
        sys.exit(f"No runs with frames.jsonl + precomputed_cache_dir under '{results_dir}'.")

    print(f"\nAvailable runs in '{results_dir}':")
    for i, d in enumerate(dirs, 1):
        meta = _load_metadata(d) or {}
        notes = meta.get("notes") or meta.get("defense_type") or ""
        print(f"  [{i}] {d.name}  — {notes}")
    print()

    n = len(dirs)
    while True:
        raw = input(f"Choose a run [1-{n}]: ").strip()
        try:
            c = int(raw)
        except ValueError:
            print(f"  Please enter a number between 1 and {n}.")
            continue
        if 1 <= c <= n:
            return dirs[c - 1]
        print(f"  Please enter a number between 1 and {n}.")


# ---------------------------------------------------------------------------
# Scene grouping + per-scene metrics
# ---------------------------------------------------------------------------

def _group_frames_by_scene(frames_path: pathlib.Path) -> dict[str, list[dict]]:
    """Return {sequence_id: [frame_row, ...]} ordered by frame_index_in_scene."""
    scenes: dict[str, list[dict]] = {}
    with open(frames_path) as f:
        for line in f:
            fr = json.loads(line)
            scenes.setdefault(fr.get("sequence_id", ""), []).append(fr)
    for rows in scenes.values():
        rows.sort(key=lambda r: r.get("frame_index_in_scene", 0))
    return scenes


def _scene_metrics(rows: list[dict]) -> dict:
    """Standard defense metrics: positive = is_attacked, pred = is_attack_detected."""
    pairs: list[tuple[bool, bool]] = []
    for fr in rows:
        dr = fr.get("defense_result")
        if dr is None:
            continue
        pairs.append((bool(fr["is_attacked"]), bool(dr["is_attack_detected"])))
    return _defense_metrics_from_pairs(pairs)


def _print_scene_table(
    ordered_seq_ids: list[str],
    scenes: dict[str, list[dict]],
    scene_names: dict[str, str],
) -> list[dict]:
    """Print the scene table and return per-scene metric summaries."""
    summaries: list[dict] = []
    print(f"\n{'#':>3}  {'scene':<12} {'sequence_id':<34} "
          f"{'frames':>6} {'atk':>5} {'prec':>6} {'rec':>6} {'f1':>6}")
    print("-" * 92)
    for i, seq in enumerate(ordered_seq_ids, 1):
        rows = scenes[seq]
        m = _scene_metrics(rows)
        n_attacked = sum(1 for fr in rows if fr["is_attacked"])
        name = scene_names.get(seq, "?")
        print(f"{i:>3}  {name:<12} {seq:<34} "
              f"{len(rows):>6} {n_attacked:>5} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
        summaries.append({"seq": seq, "name": name, "metrics": m})
    print()
    return summaries


def _pick_scenes(
    ordered_seq_ids: list[str], scene_names: dict[str, str]
) -> list[str]:
    n = len(ordered_seq_ids)
    print(f"  [1] all")
    while True:
        raw = input(f"Choose [1] for all, or space-separated [2-{n + 1}]: ").strip()
        parts = raw.split()
        try:
            choices = [int(x) for x in parts]
        except ValueError:
            print(f"  Please enter numbers between 1 and {n + 1}.")
            continue
        if parts and all(1 <= c <= n + 1 for c in choices):
            if 1 in choices:
                return ordered_seq_ids
            seen: set[int] = set()
            ordered = [c for c in choices if not (c in seen or seen.add(c))]
            return [ordered_seq_ids[c - 2] for c in ordered]
        print(f"  Please enter numbers between 1 and {n + 1}.")


def _resolve_scene_selection(
    arg_scene: str | None,
    ordered_seq_ids: list[str],
    scene_names: dict[str, str],
) -> list[str]:
    """Map a --scene CLI value (comma-separated names/sequence_ids) to seq ids."""
    if not arg_scene:
        return _pick_scenes(ordered_seq_ids, scene_names)
    name_to_seq = {v: k for k, v in scene_names.items()}
    selected: list[str] = []
    for tok in arg_scene.split(","):
        tok = tok.strip()
        if tok in ordered_seq_ids:
            selected.append(tok)
        elif tok in name_to_seq:
            selected.append(name_to_seq[tok])
        else:
            sys.exit(f"Scene '{tok}' not found. Available: {sorted(scene_names.values())}")
    return selected


# ---------------------------------------------------------------------------
# NuScenes clean-cloud lookup
# ---------------------------------------------------------------------------

def _build_nusc(dataroot: str, version: str):
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def _lidar_sd_for_frame(nusc, frame_id: str):
    """Return the LIDAR_TOP sample_data record whose token starts with frame_id."""
    matches = [sd for sd in nusc.sample_data
               if sd["token"].startswith(frame_id) and sd["channel"] == "LIDAR_TOP"]
    if not matches:
        return None
    return matches[0]


def _load_clean_lidar(nusc, dataroot: str, lidar_sd) -> np.ndarray:
    path = pathlib.Path(dataroot) / lidar_sd["filename"]
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]


def _scene_name_map(nusc, seq_ids: list[str]) -> dict[str, str]:
    """Map nuScenes scene tokens (sequence_id) to scene names."""
    out: dict[str, str] = {}
    for seq in seq_ids:
        try:
            out[seq] = nusc.get("scene", seq)["name"]
        except Exception:
            out[seq] = "?"
    return out


# ---------------------------------------------------------------------------
# Stable trace names + viewer (with legend-visibility persistence)
# ---------------------------------------------------------------------------

def _stabilise_trace_names(fig: dict) -> None:
    """Strip the per-frame ' (N pts)' suffix from diff trace names in place.

    Stable names let the viewer carry legend on/off state across frames by name.
    The counts are still available in the frame label and per-point hover text.
    """
    for trace in fig.get("data", []):
        name = trace.get("name")
        if isinstance(name, str) and " (" in name:
            trace["name"] = name.split(" (", 1)[0]


def _cluster_centroid_traces(defense_result: dict | None, centroid_threshold: float):
    """Scatter3d traces for radial_jitter clusters, as circles at their centroids.

    Tested clusters (sigma_centroid not None) are coloured yellow (0) → red (at or
    above ``centroid_threshold``).  Untested clusters (no sigma yet) are grey.
    Returns [] when there are no cluster_details (non-radial_jitter defenses).
    """
    import plotly.graph_objects as go

    meta = (defense_result or {}).get("metadata") or {}
    clusters = meta.get("cluster_details")
    if not clusters:
        return []

    tested = [c for c in clusters if c.get("sigma_centroid") is not None]
    untested = [c for c in clusters if c.get("sigma_centroid") is None]
    traces = []

    def _hover(c):
        return (f"σ_centroid={c.get('sigma_centroid')}<br>"
                f"σ_point={c.get('sigma_point')}<br>"
                f"n_points={c.get('n_points_cur')}<br>"
                f"flagged={c.get('flagged')}")

    if tested:
        c0 = np.array([c["centroid"] for c in tested], dtype=float)
        traces.append(go.Scatter3d(
            x=c0[:, 0].tolist(), y=c0[:, 1].tolist(), z=c0[:, 2].tolist(),
            mode="markers",
            marker=dict(
                size=10, symbol="circle", opacity=0.9,
                color=[float(c["sigma_centroid"]) for c in tested],
                colorscale=[[0.0, "yellow"], [1.0, "red"]],
                cmin=0.0, cmax=float(centroid_threshold),
                line=dict(color="black", width=1),
                colorbar=dict(title="σ_centroid", thickness=12, len=0.4,
                              x=1.12, y=0.25),
            ),
            name="tested clusters",
            text=[_hover(c) for c in tested],
            hovertemplate="%{text}<extra>cluster</extra>",
        ))
    if untested:
        c1 = np.array([c["centroid"] for c in untested], dtype=float)
        traces.append(go.Scatter3d(
            x=c1[:, 0].tolist(), y=c1[:, 1].tolist(), z=c1[:, 2].tolist(),
            mode="markers",
            marker=dict(size=8, symbol="circle", opacity=0.5, color="lightgray",
                        line=dict(color="black", width=1)),
            name="untested clusters",
            text=[_hover(c) for c in untested],
            hovertemplate="%{text}<extra>cluster</extra>",
        ))
    return traces


_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Attack Cache Viewer</title>
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

  // Persist the 3-D camera across frames.
  const camera = plotDiv._fullLayout?.scene?.camera;
  const layout = camera
    ? Object.assign({}, fig.layout, {scene: Object.assign({}, fig.layout.scene, {camera})})
    : fig.layout;

  // Persist legend on/off state across frames, keyed by (stable) trace name.
  const vis = {};
  (plotDiv.data || []).forEach(t => {
    if (t.name !== undefined && t.visible !== undefined) vis[t.name] = t.visible;
  });
  const newData = fig.data.map(t =>
    (t.name in vis) ? Object.assign({}, t, {visible: vis[t.name]}) : t
  );

  Plotly.react('plot', newData, layout, {responsive: true});
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
        (frames_dir / fname).write_text(
            json.dumps({"fig": fd["fig_json"]}), encoding="utf-8"
        )
        manifest.append({"file": f"frames/{fname}", "label": fd["label"]})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "viewer.html").write_text(_VIEWER_HTML, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise attacked frames of a finished run from its precomputed cache."
    )
    p.add_argument("--run-dir", default=None,
                   help="Result directory to use (skips interactive run picker)")
    p.add_argument("--precomputed-root", default=None,
                   help="Directory holding the precomputed caches "
                        "(default: parent of the metadata's precomputed_cache_dir)")
    p.add_argument("--dataroot", default=None,
                   help="NuScenes dataset root (default: metadata dataset_params.root)")
    p.add_argument("--version", default=None,
                   help="NuScenes version (default: metadata dataset_params.version)")
    p.add_argument("--scene", default=None,
                   help="Comma-separated scene names or sequence_ids to render "
                        "(skips interactive scene picker)")
    p.add_argument("--frames-per-scene", type=int, default=None,
                   help="Number of evenly-spaced frames to sample per scene (default: all)")
    p.add_argument("--output-dir", default=None,
                   help="Output dir for the viewer (default: <run-dir>/cache_attack_viz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else _pick_run_dir(RESULTS_DIR)
    meta = _load_metadata(run_dir)
    if meta is None:
        sys.exit(f"No run_metadata.json in {run_dir}.")
    frames_path = _frames_jsonl(run_dir)
    if frames_path is None:
        sys.exit(f"No *_frames.jsonl in {run_dir}.")

    print(f"Run         : {run_dir.name}  — {meta.get('notes', '')}")
    print(f"Attack type : {meta.get('attack_type')}")
    print(f"Defense type: {meta.get('defense_type')}")

    # --- Resolve cache ---
    precomputed_root = pathlib.Path(args.precomputed_root) if args.precomputed_root else None
    cache_stem = _resolve_cache_stem(meta, precomputed_root)
    print(f"Cache       : {cache_stem}")

    # --- Dataset ---
    ds_params = meta.get("dataset_params") or {}
    dataroot = args.dataroot or ds_params.get("root")
    version = args.version or ds_params.get("version")
    if not dataroot or not version:
        sys.exit("Dataset root/version unknown; pass --dataroot and --version.")
    print(f"Dataset     : {version} @ {dataroot}")
    nusc = _build_nusc(dataroot, version)

    # --- Scenes + metrics ---
    scenes = _group_frames_by_scene(frames_path)
    ordered_seq_ids = list(scenes.keys())
    scene_names = _scene_name_map(nusc, ordered_seq_ids)
    _print_scene_table(ordered_seq_ids, scenes, scene_names)

    selected = _resolve_scene_selection(args.scene, ordered_seq_ids, scene_names)
    print(f"Rendering   : {[scene_names.get(s, s) for s in selected]}")

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else run_dir / "cache_attack_viz"

    # --- Render ---
    collected: list[dict] = []
    attack_type = (meta.get("attack_type") or "attack")
    centroid_threshold = (meta.get("defense_params") or {}).get("centroid_threshold")
    with shelve.open(str(cache_stem), flag="r") as cache:
        for seq in selected:
            name = scene_names.get(seq, seq)
            rows = scenes[seq]
            attacked_rows = [fr for fr in rows if fr["is_attacked"]]
            sampled = _evenly_spaced(attacked_rows, args.frames_per_scene)
            print(f"\n  {name} ({seq})  "
                  f"{len(attacked_rows)} attacked → {len(sampled)} sampled")

            for fi, fr in enumerate(sampled):
                frame_id = fr["frame_id"]
                entry = cache.get(frame_id)
                if entry is None or getattr(entry, "attacked_lidar", None) is None:
                    print(f"    [skip] {frame_id}: no attacked_lidar in cache")
                    continue
                lidar_sd = _lidar_sd_for_frame(nusc, frame_id)
                if lidar_sd is None:
                    print(f"    [skip] {frame_id}: no LIDAR_TOP sample_data in dataset")
                    continue

                clean = _load_clean_lidar(nusc, dataroot, lidar_sd)
                attacked = entry.attacked_lidar.astype(np.float32)
                ghost_mask, removed_mask = _find_diff_masks(clean, attacked)
                ghost_pts = attacked[ghost_mask]
                removed_pts = clean[removed_mask]

                fig = _make_diff_figure(
                    clean=clean,
                    ghost_pts=ghost_pts,
                    removed_pts=removed_pts,
                    scene=name,
                    filename=frame_id,
                    frame_idx=fi,
                    total_frames=len(sampled),
                    attack_type=attack_type,
                )
                # Overlay detector prediction boxes (sensor frame).
                box_traces = _bbox_traces(entry.clean_predictions or [], "clean preds", "cyan")
                if entry.attacked_predictions:
                    box_traces += _bbox_traces(entry.attacked_predictions, "attacked preds", "orange")
                for t in box_traces:
                    fig["data"].append(t.to_plotly_json())

                # Defense output overlay: radial_jitter cluster circles coloured
                # by sigma_centroid.  Threshold falls back to the per-frame value.
                dr = fr.get("defense_result") or {}
                thr = centroid_threshold
                if thr is None:
                    thr = (dr.get("metadata") or {}).get("centroid_threshold")
                if thr is not None:
                    for t in _cluster_centroid_traces(dr, thr):
                        fig["data"].append(t.to_plotly_json())

                _stabilise_trace_names(fig)

                meta_fr = fr.get("attack_metadata") or {}
                n_inj = meta_fr.get("n_injected")
                label = (
                    f"{name} | frame {fi + 1}/{len(sampled)} ({frame_id}) | "
                    f"attacked=T successful={fr.get('attack_successful')} "
                    f"detected={fr['defense_result']['is_attack_detected']}"
                    + (f" | n_injected={n_inj}" if n_inj is not None else "")
                )
                collected.append({"fig_json": fig, "label": label})
                print(f"    [{fi + 1:2d}/{len(sampled)}] clean={len(clean):,} "
                      f"injected={len(ghost_pts):,} removed={len(removed_pts):,}")

    if not collected:
        sys.exit("No renderable attacked frames found for the selected scene(s).")

    print(f"\nWriting {len(collected)} frames → {out_dir}")
    _write_output(collected, out_dir)
    print(f"Viewer ready.  Serve with:\n  python -m http.server --directory {out_dir}\n"
          "then open http://localhost:8000/viewer.html")


if __name__ == "__main__":
    main()
