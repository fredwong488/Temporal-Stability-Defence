"""
tools/test_llm_defense.py
--------------------------
Smoke-test for LLMDefense: runs the defense on N attacked keyframes from
NuScenes, saves the three rendered views (BEV, isometric, camera) as PNGs, and
writes an HTML report showing each view alongside the LLM's structured response.

The script loads clean NuScenes keyframes and applies GhostObjectAttack directly
using the pre-recorded ghost cloud traces in eval_pipeline/attacks/ghost_object/traces/.

Usage (run from the project root):
    pixi run python tools/test_llm_defense.py \\
        --nuscenes-root /vol/bitbucket/<user>/data/nuscenes \\
        --attack-type car \\
        --n-frames 3 \\
        --output-dir /tmp/llm_defense_test

    # Optional overrides:
    --backend gemini
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

_TRACES_DIR = _ROOT / "eval_pipeline" / "attacks" / "ghost_object" / "traces"

_GHOST_CLOUD_FILE = {
    "car": _TRACES_DIR / "ghost_cloud_car.npy",
    "cyl": _TRACES_DIR / "ghost_cloud_cyl.npy",
    "ped": _TRACES_DIR / "ghost_cloud_ped.npy",
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
     total_elapsed: {total_elapsed_s:.2f}s &nbsp;|&nbsp;
     query_elapsed: {query_elapsed_s:.2f}s &nbsp;|&nbsp;
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

def _find_clean_keyframes(nusc, lidar_channel: str, n_frames: int) -> list[str]:
    """Return up to n_frames sample_data tokens for keyframes on the given LiDAR channel."""
    results: list[str] = []
    for sd in nusc.sample_data:
        if sd["channel"] == lidar_channel and sd["is_key_frame"]:
            results.append(sd["token"])
            if len(results) >= n_frames:
                break
    return results


def _load_clean_frame(nusc, nuscenes_root: pathlib.Path, sd_token: str) -> "Frame":
    """Load a clean NuScenes Frame for the given sample_data token."""
    from eval_pipeline.datasets.nuscenes import _annotation_to_label, _load_lidar, _sensor_to_global
    from eval_pipeline.types import Frame

    sd = nusc.get("sample_data", sd_token)
    ego_pose, sensor_to_ego = _sensor_to_global(nusc, sd)
    timestamp = sd["timestamp"] / 1e6

    lidar = _load_lidar(nuscenes_root, sd)

    sample = nusc.get("sample", sd["sample_token"])
    labels = []
    for ann_token in sample["anns"]:
        lbl = _annotation_to_label(nusc, ann_token, ego_pose)
        if lbl is not None:
            labels.append(lbl)

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
        lidar=lidar,
        image=image,
        labels=labels,
        kitti_calib=None,
        nuscenes_ego_pose=ego_pose,
        nuscenes_sensor_to_ego=sensor_to_ego,
        is_attacked=False,
        attacked_modalities=frozenset(),
        attack_metadata={},
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
        default="/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets/nuscenes-v1.0-mini",
        help="Path to NuScenes dataset root",
    )
    p.add_argument(
        "--nuscenes-version",
        default="v1.0-mini",
        help="NuScenes version string (default: v1.0-mini)",
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
        help="Number of frames to test (default: 3)",
    )
    p.add_argument(
        "--backend",
        choices=["gemini"],
        default="gemini",
        help="LLM backend (default: gemini)",
    )
    p.add_argument(
        "--prompt-path",
        default="eval_pipeline/defenses/llm/llm_prompt.md",
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
    p.add_argument(
        "--detector",
        choices=["pointpillars_nuscenes", "none"],
        default="pointpillars_nuscenes",
        help="Detector to overlay bounding boxes on rendered views (default: pointpillars_nuscenes)",
    )
    p.add_argument("--roi-forward", type=float, default=50.0, help="ROI forward extent in metres (default: 50.0)")
    p.add_argument("--roi-side", type=float, default=20.0, help="ROI lateral half-width in metres (default: 20.0)")
    p.add_argument("--roi-rear", type=float, default=50.0, help="ROI rear extent in metres (default: 50.0)")
    p.add_argument("--ego-front", type=float, default=2.0, help="Ego-box forward extent in metres (default: 2.0)")
    p.add_argument("--ego-rear", type=float, default=2.0, help="Ego-box rear extent in metres (default: 2.0)")
    p.add_argument("--ego-side", type=float, default=1.4, help="Ego-box half-width in metres (default: 1.4)")
    p.add_argument("--render-dpi", type=int, default=150, help="DPI for rendered view images (default: 150)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    args = parse_args()

    nuscenes_root = pathlib.Path(args.nuscenes_root)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ghost_cloud_file = _GHOST_CLOUD_FILE[args.attack_type]
    ghost_cloud_path = _TRACES_DIR / ghost_cloud_file

    if not nuscenes_root.exists():
        sys.exit(f"NuScenes root not found: {nuscenes_root}")
    if not ghost_cloud_path.exists():
        sys.exit(
            f"Ghost cloud trace not found: {ghost_cloud_path}\n"
            f"Generate it with: pixi run python tools/visualise_ghost_attack.py --attack-type {args.attack_type}"
        )

    print(f"NuScenes root    : {nuscenes_root}")
    print(f"Ghost cloud path : {ghost_cloud_path}")
    print(f"Backend          : {args.backend}")
    print(f"Output           : {out_dir}")
    print()

    print("Loading NuScenes index…")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.nuscenes_version, dataroot=str(nuscenes_root), verbose=False)

    sd_tokens = _find_clean_keyframes(nusc, args.lidar_channel, args.n_frames)
    if not sd_tokens:
        sys.exit("No keyframes found in the NuScenes index.")
    print(f"Found {len(sd_tokens)} keyframes (requested {args.n_frames})\n")

    from eval_pipeline.attacks.ghost_object.ghost_object import GhostObjectAttack
    from eval_pipeline.defenses.llm.defense import LLMDefense
    from eval_pipeline.runner import _detector_registry
    from eval_pipeline.types import FrameHistory
    from eval_pipeline.visualisation.render_views import render_three_views

    attack = GhostObjectAttack(ghost_cloud_file=ghost_cloud_file)

    detector = None
    if args.detector != "none":
        det_cls = _detector_registry()[args.detector]
        detector = det_cls()
        print(f"Detector : {detector.name}\n")

    roi_min = (-args.roi_rear, -args.roi_side)
    roi_max = (args.roi_forward, args.roi_side)

    defense = LLMDefense(
        backend=args.backend,
        prompt_path=args.prompt_path,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
        requests_per_minute=60,
        roi_forward=args.roi_forward,
        roi_side=args.roi_side,
        roi_rear=args.roi_rear,
        ego_front=args.ego_front,
        ego_rear=args.ego_rear,
        ego_side=args.ego_side,
        render_dpi=args.render_dpi,
    )
    print(f"Defense : {defense.name}")
    print(f"async_detect : {defense.async_detect}\n")

    # Load clean frames, apply attack, render views
    loaded: list[tuple] = []  # (frame, views)
    for idx, sd_token in enumerate(sd_tokens, start=1):
        print(f"[{idx}/{len(sd_tokens)}] Loading frame {sd_token[:16]}")
        clean_frame = _load_clean_frame(nusc, nuscenes_root, sd_token)
        frame = attack.apply(clean_frame)
        predictions = detector.predict(frame) if detector is not None else []
        frame = frame.with_predictions(predictions)
        from eval_pipeline.defenses._multiframe_common import remove_ego_box
        import dataclasses
        render_frame = dataclasses.replace(frame, lidar=remove_ego_box(frame.lidar, args.ego_front, args.ego_rear, args.ego_side))
        views = render_three_views(render_frame, predictions=predictions, roi_min=roi_min, roi_max=roi_max, dpi=args.render_dpi)
        for view_name, png_bytes in views.items():
            png_path = out_dir / f"frame_{idx:02d}_{view_name}.png"
            png_path.write_bytes(png_bytes)
            print(f"  Saved {png_path.name}")
        loaded.append((frame, views))
    print()

    # Submit defense calls exactly as the pipeline does:
    # - ThreadPoolExecutor when async_detect is True
    # - direct call otherwise
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
            f"  cache_hit={result.metadata.get('cache_hit')}"
            f"  total={result.metadata.get('total_elapsed_s', 0):.2f}s"
            f"  query={result.metadata.get('query_elapsed_s', 0):.2f}s"
        )
        frames_html_parts.append(_FRAME_TEMPLATE.format(
            idx=idx,
            frame_id=frame.frame_id,
            sequence_id=frame.sequence_id,
            is_attacked=frame.is_attacked,
            total_elapsed_s=result.metadata.get("total_elapsed_s", 0),
            query_elapsed_s=result.metadata.get("query_elapsed_s", 0),
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
