"""
tools/check_sweep_cadence.py
----------------------------
Measure the empirical LIDAR_TOP cadence in a NuScenes scene by walking the
sample_data prev/next chain and reporting inter-sweep timestamp deltas.

Answers: are LiDAR sample_data frames 10 Hz or 20 Hz, and how far back in
seconds does a 10-sweep accumulation actually reach?

Usage:
    pixi run python tools/check_sweep_cadence.py \
        --dataroot data/datasets/nuscenes-v1.0-mini --version v1.0-mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        default=str(Path(__file__).parent.parent / "data/datasets/nuscenes-v1.0-mini"),
    )
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--channel", default="LIDAR_TOP")
    parser.add_argument("--n-sweeps", type=int, default=10,
                        help="accumulation depth to report span for")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[0]
    print(f"Scene: {scene['name']}  ({scene['token']})")

    # Rewind to first sweep of the channel
    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", first_sample["data"][args.channel])
    while sd["prev"] != "":
        sd = nusc.get("sample_data", sd["prev"])

    ts: list[float] = []
    is_key: list[bool] = []
    while True:
        ts.append(sd["timestamp"] / 1e6)  # us -> s
        is_key.append(bool(sd["is_key_frame"]))
        if sd["next"] == "":
            break
        sd = nusc.get("sample_data", sd["next"])

    ts_arr = np.array(ts)
    deltas = np.diff(ts_arr) * 1000.0  # ms

    n_key = sum(is_key)
    print(f"Total {args.channel} sample_data frames in scene: {len(ts)}")
    print(f"  keyframes (annotated): {n_key}")
    print(f"  intermediate sweeps  : {len(ts) - n_key}")
    print()
    print(f"Inter-frame delta (ms): mean={deltas.mean():.1f}  "
          f"median={np.median(deltas):.1f}  min={deltas.min():.1f}  max={deltas.max():.1f}")
    print(f"Implied rate: {1000.0 / np.median(deltas):.1f} Hz")
    print()

    # Span covered by an N-sweep accumulation (current + N-1 previous)
    if len(ts) >= args.n_sweeps:
        span = ts_arr[args.n_sweeps - 1] - ts_arr[0]
        print(f"{args.n_sweeps}-sweep accumulation spans ~{span * 1000:.0f} ms "
              f"({span:.3f} s) of history.")

    # Keyframe-to-keyframe spacing
    key_ts = ts_arr[np.array(is_key)]
    if len(key_ts) >= 2:
        key_dt = np.diff(key_ts) * 1000.0
        print(f"Keyframe spacing (ms): median={np.median(key_dt):.1f} "
              f"-> {1000.0 / np.median(key_dt):.1f} Hz")


if __name__ == "__main__":
    main()
