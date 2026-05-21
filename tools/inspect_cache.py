"""
inspect_cache.py
----------------
Unpickle and summarise a precomputed pipeline cache file.

Usage:
    pixi run python tools/inspect_cache.py <cache.pkl> [--frame <frame_id>]
"""

import argparse
import pickle
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a precomputed pipeline cache file.")
    parser.add_argument("cache", help="Path to the .pkl cache file")
    parser.add_argument("--frame", help="Print full details for a specific frame_id")
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        cache: dict = pickle.load(f)

    print(f"Cache file : {args.cache}")
    print(f"Total frames: {len(cache)}")

    if not cache:
        return

    # Summarise field presence across all entries
    sample = next(iter(cache.values()))
    fields = list(vars(sample).keys()) if hasattr(sample, "__dict__") else []

    attacked = sum(1 for e in cache.values() if e.is_attacked)
    has_lidar = sum(1 for e in cache.values() if getattr(e, "attacked_lidar", None) is not None)
    has_atk_preds = sum(1 for e in cache.values() if e.attacked_predictions is not None)

    print(f"Attacked frames     : {attacked} / {len(cache)}")
    print(f"With attacked_lidar : {has_lidar} / {attacked} attacked")
    print(f"With attacked_preds : {has_atk_preds} / {attacked} attacked")
    print(f"Fields in entry     : {fields}")

    if args.frame:
        entry = cache.get(args.frame)
        if entry is None:
            print(f"\nFrame '{args.frame}' not found in cache.", file=sys.stderr)
            sys.exit(1)
        print(f"\n--- Frame: {args.frame} ---")
        print(f"  is_attacked        : {entry.is_attacked}")
        print(f"  attack_metadata    : {entry.attack_metadata}")
        print(f"  clean_predictions  : {len(entry.clean_predictions)} preds")
        print(f"  attacked_predictions: {len(entry.attacked_predictions) if entry.attacked_predictions is not None else 'None'}")
        lidar = getattr(entry, "attacked_lidar", None)
        if lidar is not None:
            print(f"  attacked_lidar     : shape={lidar.shape}, dtype={lidar.dtype}")
        else:
            print(f"  attacked_lidar     : None")
    else:
        # Print a one-line summary per frame
        print("\nframe_id                          | attacked | lidar | #clean | #attacked")
        print("-" * 75)
        for fid, entry in list(cache.items()):
            lidar = getattr(entry, "attacked_lidar", None)
            n_clean = len(entry.clean_predictions)
            n_atk = len(entry.attacked_predictions) if entry.attacked_predictions is not None else "-"
            print(
                f"{fid:<33} | {'yes' if entry.is_attacked else 'no ':3}    "
                f"| {'yes' if lidar is not None else 'no ':3}  "
                f"| {n_clean:6} | {n_atk}"
            )


if __name__ == "__main__":
    main()
