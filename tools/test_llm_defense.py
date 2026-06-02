"""
tools/test_llm_defense.py
--------------------------
Smoke-test for LLMDefense: runs the defense on 3 attacked keyframes from
NuScenes, saves the three rendered views (BEV, isometric, camera) as PNGs, and
writes an HTML report showing each view alongside the LLM's structured response.

The script finds attacked frames by looking for the ghost-attack LiDAR sweeps
at <nuscenes-root>/sweeps/LIDAR_TOP_GHOST_ATTACK/<CAR|CYL|PED>/ and matching
them to the corresponding NuScenes keyframes so that full metadata (ego_pose,
image path, etc.) is available.

Usage (run from the project root):
    pixi run python tools/test_llm_defense.py \\
        --nuscenes-root /vol/bitbucket/<user>/data/nuscenes \\
        --attack-type car \\
        --n-frames 3 \\
        --output-dir /tmp/llm_defense_test

    # Optional overrides:
    --backend gemini          # or qwen
    --prompt-path notes/llm_prompts.md
    --cache-dir cache/llm_defense
    --force-refresh           # ignore cached LLM responses
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import logging
import pathlib
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

_ATTACK_SUBDIR = {
    "car": "LIDAR_TOP_ATTACK_CAR",
    "cyl": "LIDAR_TOP_ATTACK_CYL",
    "ped": "LIDAR_TOP_ATTACK_PED",
}

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LLM Defense Test</title>
<style>
  body {{ background: #0d1117; color: #cdd9e5; font-family: monospace;
          margin: 0; padding: 16px; }}
  h1   {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  h2   {{ color: #79c0ff; margin-top: 32px; }}
  .frame-block {{ border: 1px solid #30363d; border-radius: 6px;
                  padding: 16px; margin-bottom: 32px; }}
  .views {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
  .views figure {{ margin: 0; }}
  .views figcaption {{ text-align: center; color: #8b949e; font-size: 0.85em;
                       margin-top: 4px; }}
  .views img {{ max-width: 460px; border: 1px solid #30363d; border-radius: 4px;
                display: block; }}
  .verdict {{ font-size: 1.1em; font-weight: bold; margin-bottom: 8px; }}
  .verdict.attack    {{ color: #f85149; }}
  .verdict.benign    {{ color: #3fb950; }}
  .verdict.uncertain {{ color: #e3b341; }}
  pre {{ background: #161b22; border: 1px solid #30363d; border-radius: 4px;
         padding: 12px; overflow-x: auto; white-space: pre-wrap; font-size: 0.85em; }}
  .meta {{ color: #8b949e; font-size: 0.85em; margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>LLM Defense Smoke-Test</h1>
<p class="meta">Backend: {backend} &nbsp;|&nbsp; Model: {model} &nbsp;|&nbsp;
Attack type: {attack_type} &nbsp;|&nbsp; Generated: {ts}</p>
{frames_html}
</body>
</html>
"""

_FRAME_TEMPLATE = """\
<div class="frame-block">
  <h2>Frame {idx} &mdash; {frame_id}</h2>
  <p class="meta">Scene: {sequence_id} &nbsp;|&nbsp;
     is_attacked: {is_attacked} &nbsp;|&nbsp;
     elapsed: {elapsed_s:.2f}s &nbsp;|&nbsp;
     cache_hit: {cache_hit}</p>
  <div class="views">
    <figure><img src="data:image/png;base64,{bev_b64}"><figcaption>BEV</figcaption></figure>
    <figure><img src="data:image/png;base64,{iso_b64}"><figcaption>Isometric</figcaption></figure>
    <figure><img src="data:image/png;base64,{cam_b64}"><figcaption>Camera</figcaption></figure>
  </div>
  <div class="verdict {verdict_cls}">Verdict: {verdict}</div>
  <p>Confidence: {confidence:.2f} &nbsp;|&nbsp; attack_detected: {is_attack_detected}</p>
  <pre>{response_json}</pre>
</div>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_attacked_bin(path: pathlib.Path) -> np.ndarray:
    """Load a NuScenes .pcd.bin as (N, 4) float32 [x, y, z, intensity]."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]


def _find_attacked_keyframes(
    nusc,
    attack_dir: pathlib.Path,
    lidar_channel: str,
    n_frames: int,
) -> list[tuple[str, pathlib.Path]]:
    """Return up to n_frames (sd_token, attack_path) pairs for keyframes only.

    Attacked sweeps live in attack_dir.  We look up each attacked filename in
    the NuScenes sample_data index and keep only keyframes so that annotations
    and camera images are available.
    """
    filename_to_sd: dict[str, dict] = {}
    for sd in nusc.sample_data:
        if sd["channel"] == lidar_channel:
            fname = pathlib.Path(sd["filename"]).name
            filename_to_sd[fname] = sd

    results: list[tuple[str, pathlib.Path]] = []
    for attack_path in sorted(attack_dir.glob("*.pcd.bin")):
        sd = filename_to_sd.get(attack_path.name)
        if sd is None or not sd["is_key_frame"]:
            continue
        results.append((sd["token"], attack_path))
        if len(results) >= n_frames:
            break

    return results


def _load_frame_with_attacked_lidar(
    nusc,
    nuscenes_root: pathlib.Path,
    sd_token: str,
    attack_path: pathlib.Path,
) -> "Frame":
    """Load a NuScenes Frame with the attacked LiDAR substituted in."""
    from eval_pipeline.datasets.nuscenes import _annotation_to_label, _sensor_to_global
    from eval_pipeline.types import Frame

    sd = nusc.get("sample_data", sd_token)
    ego_pose, sensor_to_ego = _sensor_to_global(nusc, sd)
    timestamp = sd["timestamp"] / 1e6

    attacked_lidar = _load_attacked_bin(attack_path)

    sample = nusc.get("sample", sd["sample_token"])
    labels = []
    for ann_token in sample["anns"]:
        lbl = _annotation_to_label(nusc, ann_token, ego_pose)
        if lbl is not None:
            labels.append(lbl)

    # Load CAM_FRONT image
    import cv2
    cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    cam_path = nuscenes_root / cam_sd["filename"]
    image = cv2.imread(str(cam_path))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scene = nusc.get("scene", sample["scene_token"])

    return Frame(
        frame_id=sd_token[:16],
        sequence_id=scene["token"],
        timestamp=timestamp,
        lidar=attacked_lidar,
        image=image,
        labels=labels,
        kitti_calib=None,
        nuscenes_ego_pose=ego_pose,
        nuscenes_sensor_to_ego=sensor_to_ego,
        is_attacked=True,
        attacked_modalities=frozenset({"lidar"}),
        attack_metadata={"attack": "GhostObject", "attack_file": str(attack_path)},
    )


def _verdict_css_class(verdict: str) -> str:
    if "ATTACK" in verdict:
        return "attack"
    if "BENIGN" in verdict:
        return "benign"
    return "uncertain"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test for LLMDefense.")
    p.add_argument(
        "--nuscenes-root",
        default="/vol/bitbucket/fw420/data/nuscenes",
        help="Path to NuScenes dataset root",
    )
    p.add_argument(
        "--nuscenes-version",
        default="v1.0-trainval",
        help="NuScenes version string (default: v1.0-trainval)",
    )
    p.add_argument(
        "--attack-type",
        choices=["car", "cyl", "ped"],
        default="car",
        help="Ghost attack variant to test (default: car)",
    )
    p.add_argument(
        "--n-frames",
        type=int,
        default=3,
        help="Number of attacked frames to test (default: 3)",
    )
    p.add_argument(
        "--backend",
        choices=["gemini", "qwen"],
        default="gemini",
        help="LLM backend (default: gemini)",
    )
    p.add_argument(
        "--prompt-path",
        default="notes/llm_prompts.md",
        help="Path to LLM prompt markdown file",
    )
    p.add_argument(
        "--cache-dir",
        default="cache/llm_defense",
        help="LLM response cache directory",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached LLM responses and re-query",
    )
    p.add_argument(
        "--output-dir",
        default="/tmp/llm_defense_test",
        help="Directory to write the HTML report and per-frame PNGs",
    )
    p.add_argument(
        "--lidar-channel",
        default="LIDAR_TOP",
        help="NuScenes LiDAR channel name (default: LIDAR_TOP)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    args = parse_args()

    nuscenes_root = pathlib.Path(args.nuscenes_root)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_subdir = _ATTACK_SUBDIR[args.attack_type]
    attack_dir = nuscenes_root / "sweeps" / "LIDAR_TOP_GHOST_ATTACK" / attack_subdir

    if not nuscenes_root.exists():
        sys.exit(f"NuScenes root not found: {nuscenes_root}")
    if not attack_dir.is_dir():
        sys.exit(
            f"Ghost attack directory not found: {attack_dir}\n"
            f"Expected structure: <nuscenes-root>/sweeps/LIDAR_TOP_GHOST_ATTACK/{attack_subdir}/"
        )

    print(f"NuScenes root : {nuscenes_root}")
    print(f"Attack dir    : {attack_dir}")
    print(f"Backend       : {args.backend}")
    print(f"Output        : {out_dir}")
    print()

    print("Loading NuScenes index…")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.nuscenes_version, dataroot=str(nuscenes_root), verbose=False)

    pairs = _find_attacked_keyframes(nusc, attack_dir, args.lidar_channel, args.n_frames)
    if not pairs:
        sys.exit(
            "No matching attacked keyframes found.\n"
            "Check that the attack directory contains .pcd.bin files whose filenames "
            "match LiDAR sweeps in the NuScenes index."
        )
    print(f"Found {len(pairs)} attacked keyframes (requested {args.n_frames})\n")

    from eval_pipeline.defenses.llm.defense import LLMDefense
    from eval_pipeline.types import FrameHistory
    from eval_pipeline.visualisation.render_views import render_three_views

    defense = LLMDefense(
        backend=args.backend,
        prompt_path=args.prompt_path,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
        requests_per_minute=60,
    )
    print(f"Defense : {defense.name}")
    print(f"async_detect : {defense.async_detect}\n")

    # Load all frames and render their views up front (main thread only)
    loaded: list[tuple] = []  # (frame, views)
    for idx, (sd_token, attack_path) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] Loading frame {sd_token[:16]}  ({attack_path.name})")
        frame = _load_frame_with_attacked_lidar(nusc, nuscenes_root, sd_token, attack_path)
        views = render_three_views(frame, predictions=[], dpi=120)
        for view_name, png_bytes in views.items():
            png_path = out_dir / f"frame_{idx:02d}_{view_name}.png"
            png_path.write_bytes(png_bytes)
            print(f"  Saved {png_path.name}")
        loaded.append((frame, views))
    print()

    # Submit defense calls exactly as the pipeline does:
    # - ThreadPoolExecutor when async_detect is True
    # - direct call otherwise
    # History is empty for this test (no prior frames), but we snapshot the
    # deque each iteration just as the pipeline does to avoid aliasing.
    defense_futures: list[Future | None] = []
    executor: ThreadPoolExecutor | None = (
        ThreadPoolExecutor() if defense.async_detect else None
    )

    print(f"Submitting {len(loaded)} defense call(s)…")
    t_submit = time.perf_counter()
    for frame, _views in loaded:
        history_snapshot = FrameHistory(clean=deque(), dirty=deque())
        if executor is not None:
            defense_futures.append(
                executor.submit(defense.detect, frame, history_snapshot)
            )
        else:
            defense_futures.append(None)

    # Resolve futures (pipeline pattern)
    results = []
    if executor is not None:
        executor.shutdown(wait=True)
        for i, fut in enumerate(defense_futures):
            try:
                results.append(fut.result())
            except Exception:
                logging.exception("Async defense failed for frame %s", loaded[i][0].frame_id)
                results.append(None)
    else:
        # Synchronous path: run now (futures list holds None placeholders)
        for frame, _views in loaded:
            history_snapshot = FrameHistory(clean=deque(), dirty=deque())
            results.append(defense.detect(frame, history_snapshot))

    t_total = time.perf_counter() - t_submit
    print(f"All done in {t_total:.2f}s\n")

    # Build HTML report
    frames_html_parts: list[str] = []
    for idx, ((frame, views), result) in enumerate(zip(loaded, results), start=1):
        if result is None:
            print(f"  [{idx}] ERROR — defense raised an exception (see log above)")
            continue
        verdict = result.metadata.get("verdict", "UNKNOWN")
        print(
            f"  [{idx}] frame={frame.frame_id}  verdict={verdict}"
            f"  confidence={result.confidence:.2f}"
            f"  attack_detected={result.is_attack_detected}"
            f"  cache_hit={result.metadata.get('cache_hit')}  elapsed={result.metadata.get('elapsed_s', 0):.2f}s"
        )
        frames_html_parts.append(_FRAME_TEMPLATE.format(
            idx=idx,
            frame_id=frame.frame_id,
            sequence_id=frame.sequence_id,
            is_attacked=frame.is_attacked,
            elapsed_s=result.metadata.get("elapsed_s", 0),
            cache_hit=result.metadata.get("cache_hit", False),
            bev_b64=base64.b64encode(views["bev"]).decode(),
            iso_b64=base64.b64encode(views["isometric"]).decode(),
            cam_b64=base64.b64encode(views["camera"]).decode(),
            verdict_cls=_verdict_css_class(verdict),
            verdict=verdict,
            confidence=result.confidence,
            is_attack_detected=result.is_attack_detected,
            response_json=json.dumps(result.metadata, indent=2, default=str),
        ))

    print()
    report_path = out_dir / "report.html"
    report_path.write_text(
        _HTML_TEMPLATE.format(
            backend=args.backend,
            model=defense._model,
            attack_type=args.attack_type.upper(),
            ts=datetime.datetime.now().isoformat(timespec="seconds"),
            frames_html="\n".join(frames_html_parts),
        ),
        encoding="utf-8",
    )
    print(f"Report written → {report_path}")
    print(f"Serve with:  python -m http.server --directory {out_dir}")


if __name__ == "__main__":
    main()
