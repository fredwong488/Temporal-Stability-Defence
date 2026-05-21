"""
tools/clustering_quality.py
----------------------------
Compute the clustering-quality metric for radial-jitter defenses.

For each attacked frame that has cluster_details, one global Hungarian assignment
matches cluster centroids against two target types:

  - Phantom targets  : ORA reinjected_centroid (from attack_metadata)
  - Prediction targets  : attacked_predictions box centroids (BEV containment match)

Scores reported:
  spoofed_f1  = H(spoofed_recall, global_precision)
  prediction_f1  = H(prediction_recall, global_precision)

Global precision is a shared over-clustering penalty so that clusters on predictions
are not counted as phantom false-positives.

Run:
    pixi run python tools/clustering_quality.py                # interactive picker
    pixi run python tools/clustering_quality.py results/dir1 ... # explicit run dir(s)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

RESULTS_ROOT = Path(__file__).parent.parent / "results"

SPOOF_DIST_THRESHOLD = 1.5   # metres — phantom match gate (matches count_spoofed.py)
PREDICTION_MATCH_MARGIN = 0.5   # metres buffer added to BEV polygon for prediction match
_INF = 1e9


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bev_polygon(corners_velo: list) -> Polygon:
    """corners_velo is a (8,3) nested list; returns a buffered Shapely BEV polygon."""
    pts = [(c[0], c[1]) for c in corners_velo[:4]]
    cx = sum(p[0] for p in pts) / 4
    cy = sum(p[1] for p in pts) / 4
    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
    sorted_pts = [p for _, p in sorted(zip(angles, pts))]
    return Polygon(sorted_pts).buffer(PREDICTION_MATCH_MARGIN)


# ---------------------------------------------------------------------------
# Core per-frame matching
# ---------------------------------------------------------------------------

def match_frame(
    centroids: list[tuple[float, float, float]],
    phantom_targets: list[tuple[float, float, float]],
    prediction_boxes: list[dict],  # each has "x","y","corners_velo"
) -> tuple[int, int, int, int, int, int]:
    """One global Hungarian assignment over phantoms + predictions.

    Returns (matched_phantoms, matched_predictions, matched_clusters,
             total_clusters, total_phantoms, total_predictions).
    """
    n_c, n_p, n_v = len(centroids), len(phantom_targets), len(prediction_boxes)
    n_t = n_p + n_v

    if n_c == 0:
        return 0, 0, 0, 0, n_p, n_v
    if n_t == 0:
        return 0, 0, 0, n_c, 0, 0

    cost = np.full((n_c, n_t), _INF)

    for i, (cx, cy, cz) in enumerate(centroids):
        for j, (sx, sy, sz) in enumerate(phantom_targets):
            d = math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2)
            if d <= SPOOF_DIST_THRESHOLD:
                cost[i, j] = d

        for j, box in enumerate(prediction_boxes):
            poly = _bev_polygon(box["corners_velo"])
            if poly.contains(Point(cx, cy)):
                cost[i, n_p + j] = math.sqrt((cx - box["x"]) ** 2 + (cy - box["y"]) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_phantoms = matched_predictions = matched_clusters = 0
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < _INF:
            matched_clusters += 1
            if c < n_p:
                matched_phantoms += 1
            else:
                matched_predictions += 1

    return matched_phantoms, matched_predictions, matched_clusters, n_c, n_p, n_v


# ---------------------------------------------------------------------------
# Aggregation + score computation
# ---------------------------------------------------------------------------

def compute_scores(
    matched_phantoms: int,
    matched_predictions: int,
    matched_clusters: int,
    total_clusters: int,
    total_phantoms: int,
    total_predictions: int,
    frames_scored: int = 0,
) -> dict:
    def h(a: float, b: float) -> float:
        return 2 * a * b / (a + b) if (a + b) > 0.0 else 0.0

    precision = matched_clusters / total_clusters if total_clusters > 0 else 0.0
    spoofed_recall = matched_phantoms / total_phantoms if total_phantoms > 0 else 0.0
    pred_recall = matched_predictions / total_predictions if total_predictions > 0 else 0.0

    return {
        "spoofed_f1": h(spoofed_recall, precision),
        "pred_f1": h(pred_recall, precision),
        "precision": precision,
        "spoofed_recall": spoofed_recall,
        "pred_recall": pred_recall,
        "matched_phantoms": matched_phantoms,
        "total_phantoms": total_phantoms,
        "matched_predictions": matched_predictions,
        "total_predictions": total_predictions,
        "matched_clusters": matched_clusters,
        "total_clusters": total_clusters,
        "frames_scored": frames_scored,
    }


def analyse_file(path: Path) -> dict:
    """Return raw accumulated counts for one *_frames.jsonl file."""
    acc = dict(
        matched_phantoms=0, matched_predictions=0, matched_clusters=0,
        total_clusters=0, total_phantoms=0, total_predictions=0,
        frames_scored=0,
    )

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            if not d.get("is_attacked"):
                continue

            meta = (d.get("defense_result") or {}).get("metadata") or {}
            cluster_details = meta.get("cluster_details")
            if not cluster_details:
                continue

            centroids: list[tuple[float, float, float]] = []
            for cd in cluster_details:
                c = cd.get("centroid") or []
                if len(c) == 3 and all(v is not None for v in c):
                    centroids.append((float(c[0]), float(c[1]), float(c[2])))

            phantom_targets: list[tuple[float, float, float]] = []
            for obj in (d.get("attack_metadata") or {}).get("removed_per_obj") or []:
                rc = obj.get("reinjected_centroid") or []
                if len(rc) == 3:
                    phantom_targets.append((float(rc[0]), float(rc[1]), float(rc[2])))

            pred_boxes = [
                p for p in (d.get("attacked_predictions") or [])
                if isinstance(p, dict) and p.get("corners_velo")
            ]

            mp, mv, mc, tc, n_p, n_v = match_frame(centroids, phantom_targets, pred_boxes)
            acc["matched_phantoms"] += mp
            acc["matched_predictions"] += mv
            acc["matched_clusters"] += mc
            acc["total_clusters"] += tc
            acc["total_phantoms"] += n_p
            acc["total_predictions"] += n_v
            acc["frames_scored"] += 1

    return acc


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_scores(label: str, s: dict) -> None:
    print(label)
    print(f"  frames scored : {s['frames_scored']}")
    print(f"  spoofed_f1    : {s['spoofed_f1']:.4f}  "
          f"(recall {s['spoofed_recall']:.4f}, "
          f"{s['matched_phantoms']}/{s['total_phantoms']} phantoms matched)")
    print(f"  pred_f1       : {s['pred_f1']:.4f}  "
          f"(recall {s['pred_recall']:.4f}, "
          f"{s['matched_predictions']}/{s['total_predictions']} predictions matched)")
    print(f"  precision     : {s['precision']:.4f}  "
          f"({s['matched_clusters']}/{s['total_clusters']} clusters matched a target)")


# ---------------------------------------------------------------------------
# CLI helpers (mirrors count_spoofed.py)
# ---------------------------------------------------------------------------

def list_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(d for d in results_root.iterdir() if d.is_dir() and list(d.glob("*_frames.jsonl")))


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


def main() -> None:
    if len(sys.argv) > 1:
        search_dirs = [Path(a) for a in sys.argv[1:]]
    else:
        run_dir = pick_run_dir(RESULTS_ROOT)
        print(f"\nSelected run: {run_dir}\n")
        search_dirs = [run_dir]

    files: list[Path] = []
    for d in search_dirs:
        files.extend(sorted(d.glob("*_frames.jsonl")))

    if not files:
        print("No *_frames.jsonl files found.")
        sys.exit(1)

    all_acc = dict(
        matched_phantoms=0, matched_predictions=0, matched_clusters=0,
        total_clusters=0, total_phantoms=0, total_predictions=0, frames_scored=0,
    )

    for path in files:
        acc = analyse_file(path)
        scores = compute_scores(**acc)
        print_scores(f"{path.parent.name}/{path.stem}", scores)
        print()
        for k in all_acc:
            all_acc[k] += acc[k]

    if len(files) > 1:
        agg = compute_scores(**all_acc)
        print_scores("=== AGGREGATE ===", agg)


if __name__ == "__main__":
    main()
