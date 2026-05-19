"""
tools/count_spoofed.py
----------------------
Compare attacked-object count (from removed_per_obj) vs spoofed-cluster count
(using the same 1.5 m centroid-matching logic as plot_results.py).

Run:
    pixi run python tools/count_spoofed.py                    # interactive run picker
    pixi run python tools/count_spoofed.py results/dir1 ...   # explicit run dir(s)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent.parent / "results"
SPOOF_DIST_THRESHOLD = 1.5


def list_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and list(d.glob("*_frames.jsonl"))
    )


def pick_run_dir(results_root: Path) -> Path:
    dirs = list_run_dirs(results_root)
    if not dirs:
        sys.exit(f"No run directories with *_frames.jsonl found under '{results_root}'.")

    print(f"\nAvailable runs in '{results_root}':")
    for i, d in enumerate(dirs, 1):
        n = len(list(d.glob("*_frames.jsonl")))
        print(f"  [{i}] {d.name}  ({n} experiment{'s' if n != 1 else ''})")

    while True:
        raw = input(f"\nChoose a run [1-{len(dirs)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(dirs):
            return dirs[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(dirs)}.")


def analyse_file(path: Path) -> None:
    total_attacked = 0
    total_spoofed_clusters = 0
    frames_with_attack = 0

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            removed_per_obj = (d.get("attack_metadata") or {}).get("removed_per_obj") or []
            n_attacked = len(removed_per_obj)
            total_attacked += n_attacked
            if n_attacked > 0:
                frames_with_attack += 1

            dr = d.get("defense_result") or {}
            meta = dr.get("metadata") or {}

            spoofed_centroids: list[tuple[float, float, float]] = []
            for obj in removed_per_obj:
                rc = obj.get("reinjected_centroid")
                if rc and len(rc) == 3:
                    spoofed_centroids.append((float(rc[0]), float(rc[1]), float(rc[2])))

            for cd in meta.get("cluster_details") or []:
                centroid = cd.get("centroid") or [None, None, None]
                cx, cy, cz = centroid[0], centroid[1], centroid[2]

                is_spoofed = False
                if spoofed_centroids and cx is not None and cy is not None and cz is not None:
                    min_dist = min(
                        math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2)
                        for sx, sy, sz in spoofed_centroids
                    )
                    is_spoofed = min_dist < SPOOF_DIST_THRESHOLD

                if is_spoofed:
                    total_spoofed_clusters += 1

    missing = total_attacked - total_spoofed_clusters
    pct = missing / total_attacked * 100 if total_attacked else 0.0

    print(f"{path.parent.name}/{path.stem}")
    print(f"  attacked objects (removed_per_obj entries): {total_attacked}")
    print(f"  frames with attack:                         {frames_with_attack}")
    print(f"  is_spoofed_cluster=True clusters:           {total_spoofed_clusters}")
    print(f"  missing (not matched to a cluster):         {missing}  ({pct:.1f}%)")


def main() -> None:
    if len(sys.argv) > 1:
        search_dirs = [Path(a) for a in sys.argv[1:]]
    else:
        run_dir = pick_run_dir(RESULTS_ROOT)
        print(f"\nSelected run: {run_dir}\n")
        search_dirs = [run_dir]

    files = []
    for d in search_dirs:
        files.extend(sorted(d.glob("*_frames.jsonl")))

    if not files:
        print("No *_frames.jsonl files found. Pass result directories as arguments.")
        sys.exit(1)

    for path in files:
        analyse_file(path)
        print()


if __name__ == "__main__":
    main()
