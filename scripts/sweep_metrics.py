"""
scripts/sweep_metrics.py
------------------------
Per-metric extract, log, and CSV-write helpers for run_sweep.py.
"""

from __future__ import annotations

import csv
import logging
import pathlib


# ---------------------------------------------------------------------------
# Row extractors
# ---------------------------------------------------------------------------

def extract_ap_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
    classes: list[str],
    difficulties: list[str],
) -> dict:
    """Extract AP floats for each class/difficulty into a flat dict for the CSV."""
    attacked_map = summary.get("attack_effectiveness", {}).get("attacked_map", {})
    row: dict = {sweep_param: sweep_val}
    for cls in classes:
        cls_ap = attacked_map.get(cls, {})
        for diff in difficulties:
            row[f"{cls.lower()}_ap_{diff.lower()}"] = cls_ap.get(diff, float("nan"))
    return row



def extract_detection_rate_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
    classes: list[str],
) -> dict:
    """Extract detection-rate drop into a flat dict for the CSV.

    Columns: overall + per-class clean/attacked/absolute_drop/relative_drop.
    """
    dr = summary.get("detection_rate", {})
    row: dict = {sweep_param: sweep_val}
    for key in ["overall"] + classes:
        entry = dr.get(key, {})
        prefix = f"{key}_" if key != "overall" else ""
        row[f"{prefix}detection_rate_clean"] = entry.get("detection_rate_clean", float("nan"))
        row[f"{prefix}detection_rate_attacked"] = entry.get("detection_rate_attacked", float("nan"))
        row[f"{prefix}absolute_drop"] = entry.get("absolute_drop", float("nan"))
        row[f"{prefix}relative_drop"] = entry.get("relative_drop", float("nan"))
    return row



def extract_defense_effectiveness_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract defense F1, precision, recall, TPR and FPR into a flat dict for the CSV."""
    de = summary.get("defense_effectiveness", {})
    return {
        sweep_param: sweep_val,
        "detection_f1": de.get("f1", float("nan")),
        "detection_precision": de.get("precision", float("nan")),
        "detection_recall": de.get("recall", float("nan")),
        "tpr": de.get("tpr", float("nan")),
        "fpr": de.get("fpr", float("nan")),
    }


def extract_defense_effectiveness_filtered_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract filtered defense F1, precision, recall, TPR and FPR into a flat dict."""
    de = summary.get("defense_effectiveness_filtered", {})
    return {
        sweep_param: sweep_val,
        "detection_f1": de.get("f1", float("nan")),
        "detection_precision": de.get("precision", float("nan")),
        "detection_recall": de.get("recall", float("nan")),
        "tpr": de.get("tpr", float("nan")),
        "fpr": de.get("fpr", float("nan")),
    }


def extract_attack_success_rate_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract attack success rate into a flat dict for the CSV."""
    asr = summary.get("attack_success_rate", {})
    return {
        sweep_param: sweep_val,
        "attack_success_rate": asr.get("attack_success_rate", float("nan")),
        "n_successful": asr.get("n_successful", float("nan")),
        "n_attacked_frames": asr.get("n_attacked_frames", float("nan")),
    }


def extract_clustering_quality_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract clustering-quality F1 scores into a flat dict for the CSV."""
    cq = summary.get("clustering_quality", {})
    return {
        sweep_param: sweep_val,
        "spoofed_f1": cq.get("spoofed_f1", float("nan")),
        "pred_f1": cq.get("pred_f1", float("nan")),
        "precision": cq.get("precision", float("nan")),
        "spoofed_recall": cq.get("spoofed_recall", float("nan")),
        "pred_recall": cq.get("pred_recall", float("nan")),
    }


def extract_pacts_effectiveness_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract PACTS effectiveness F1, precision and recall into a flat dict for the CSV."""
    pe = summary.get("pacts_effectiveness", {})
    return {
        sweep_param: sweep_val,
        "pacts_f1": pe.get("f1", float("nan")),
        "pacts_precision": pe.get("precision", float("nan")),
        "pacts_recall": pe.get("recall", float("nan")),
    }



def extract_llm_attack_type_accuracy_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract LLM attack-type accuracy into a flat dict for the CSV."""
    lata = summary.get("llm_attack_type_accuracy", {})
    return {
        sweep_param: sweep_val,
        "n_tp_detected": lata.get("n_tp_detected", float("nan")),
        "n_correct_type": lata.get("n_correct_type", float("nan")),
        "n_incorrect_type": lata.get("n_incorrect_type", float("nan")),
        "type_accuracy": lata.get("type_accuracy", float("nan")),
    }



def extract_llm_cost_metrics_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract LLM token stats (mean/median/std) into a flat dict for the CSV."""
    lcm = summary.get("llm_cost_metrics", {})
    row: dict = {sweep_param: sweep_val}
    for field in ("input_tokens", "output_tokens", "thoughts_token_count"):
        for stat in ("mean", "median", "std"):
            row[f"{field}_{stat}"] = lcm.get(f"{field}_{stat}", float("nan"))
    row["n_frames"] = lcm.get("n_frames", float("nan"))
    row["n_api_frames"] = lcm.get("n_api_frames", float("nan"))
    return row



def extract_timing_metrics_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Flatten timing_metrics nested stats into a flat dict for the CSV.

    Keys are ``<timing_key>_<stat>`` (e.g. ``total_mean``, ``query_median``).
    """
    tm = summary.get("timing_metrics", {})
    row: dict = {sweep_param: sweep_val, "n_frames": tm.get("n_frames", float("nan"))}
    for key, stats in tm.items():
        if key == "n_frames" or not isinstance(stats, dict):
            continue
        for stat in ("mean", "median", "std"):
            row[f"{key}_{stat}"] = stats.get(stat, float("nan"))
    return row



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_summary_metrics(
    summary: dict,
    metric_types: list[str],
    classes: list[str],
    difficulties: list[str],
) -> None:
    """Log a one-line summary for each requested metric type."""
    if "ap" in metric_types:
        attacked_map = summary.get("attack_effectiveness", {}).get("attacked_map", {})
        for cls in classes:
            cls_ap = attacked_map.get(cls, {})
            values = "  ".join(
                f"{d}={cls_ap.get(d, float('nan')):.2f}" for d in difficulties
            )
            logging.info("  %s AP  %s", cls, values)

    if "detection_rate" in metric_types:
        overall = summary.get("detection_rate", {}).get("overall", {})
        logging.info(
            "  DetRate  clean=%.3f  attacked=%.3f  abs_drop=%.3f  rel_drop=%.1f%%",
            overall.get("detection_rate_clean", float("nan")),
            overall.get("detection_rate_attacked", float("nan")),
            overall.get("absolute_drop", float("nan")),
            (overall.get("relative_drop", float("nan")) or 0.0) * 100,
        )

    if "defense_effectiveness" in metric_types:
        de = summary.get("defense_effectiveness", {})
        logging.info(
            "  Detection  F1=%.3f  precision=%.3f  recall=%.3f  TPR=%.3f  FPR=%.3f",
            de.get("f1", float("nan")),
            de.get("precision", float("nan")),
            de.get("recall", float("nan")),
            de.get("tpr", float("nan")),
            de.get("fpr", float("nan")),
        )

    if "defense_effectiveness_filtered" in metric_types:
        de = summary.get("defense_effectiveness_filtered", {})
        logging.info(
            "  Detection (filtered)  F1=%.3f  precision=%.3f  recall=%.3f  TPR=%.3f  FPR=%.3f",
            de.get("f1", float("nan")),
            de.get("precision", float("nan")),
            de.get("recall", float("nan")),
            de.get("tpr", float("nan")),
            de.get("fpr", float("nan")),
        )

    if "attack_success_rate" in metric_types:
        asr = summary.get("attack_success_rate", {})
        logging.info(
            "  Attack success rate  %.3f  (%s/%s frames)",
            asr.get("attack_success_rate", float("nan")),
            asr.get("n_successful", "?"),
            asr.get("n_attacked_frames", "?"),
        )

    if "clustering_quality" in metric_types:
        cq = summary.get("clustering_quality", {})
        logging.info(
            "  Clustering  spoofed_f1=%.3f  pred_f1=%.3f  precision=%.3f",
            cq.get("spoofed_f1", float("nan")),
            cq.get("pred_f1", float("nan")),
            cq.get("precision", float("nan")),
        )

    if "pacts_effectiveness" in metric_types:
        pe = summary.get("pacts_effectiveness", {})
        logging.info(
            "  PACTS  F1=%.3f  precision=%.3f  recall=%.3f",
            pe.get("f1", float("nan")),
            pe.get("precision", float("nan")),
            pe.get("recall", float("nan")),
        )

    if "llm_attack_type_accuracy" in metric_types:
        lata = summary.get("llm_attack_type_accuracy", {})
        logging.info(
            "  LLM type accuracy  type_accuracy=%.3f  correct=%s/%s  incorrect=%s",
            lata.get("type_accuracy", float("nan")),
            lata.get("n_correct_type", "?"),
            lata.get("n_tp_detected", "?"),
            lata.get("n_incorrect_type", "?"),
        )

    if "llm_cost_metrics" in metric_types:
        lcm = summary.get("llm_cost_metrics", {})
        logging.info(
            "  LLM cost  in_tok=%.0f(med=%.0f)±%.0f  out_tok=%.0f(med=%.0f)±%.0f  think_tok=%.0f(med=%.0f)±%.0f  n=%s  n_api=%s",
            lcm.get("input_tokens_mean", float("nan")),
            lcm.get("input_tokens_median", float("nan")),
            lcm.get("input_tokens_std", float("nan")),
            lcm.get("output_tokens_mean", float("nan")),
            lcm.get("output_tokens_median", float("nan")),
            lcm.get("output_tokens_std", float("nan")),
            lcm.get("thoughts_token_count_mean", float("nan")),
            lcm.get("thoughts_token_count_median", float("nan")),
            lcm.get("thoughts_token_count_std", float("nan")),
            lcm.get("n_frames", "?"),
            lcm.get("n_api_frames", "?"),
        )

    if "timing_metrics" in metric_types:
        tm = summary.get("timing_metrics", {})
        total = tm.get("total", {})
        logging.info(
            "  Timing  total=%.3f(med=%.3f)±%.3f s  n=%s",
            total.get("mean", float("nan")),
            total.get("median", float("nan")),
            total.get("std", float("nan")),
            tm.get("n_frames", "?"),
        )


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_ap_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
    classes: list[str],
    difficulties: list[str],
) -> None:
    fieldnames = [sweep_param] + [
        f"{cls.lower()}_ap_{d.lower()}"
        for cls in classes
        for d in difficulties
    ]
    out_path = run_dir / f"sweep_{sweep_tag}_ap.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("AP CSV written to %s", out_path)
    print(f"\nAP results: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        vals = [str(row[sweep_param])] + [
            f"{row[f'{cls.lower()}_ap_{d.lower()}']:.4f}"
            for cls in classes
            for d in difficulties
        ]
        print(",".join(vals))


def write_detection_rate_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
    classes: list[str],
) -> None:
    fieldnames = [sweep_param]
    for key in ["overall"] + classes:
        prefix = f"{key}_" if key != "overall" else ""
        fieldnames += [
            f"{prefix}detection_rate_clean",
            f"{prefix}detection_rate_attacked",
            f"{prefix}absolute_drop",
            f"{prefix}relative_drop",
        ]
    out_path = run_dir / f"sweep_{sweep_tag}_detection_rate.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Detection rate CSV written to %s", out_path)
    print(f"\nDetection rate: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        vals = [str(row[sweep_param])] + [
            f"{row.get(col, float('nan')):.4f}" for col in fieldnames[1:]
        ]
        print(",".join(vals))


def write_defense_effectiveness_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    fieldnames = [sweep_param, "detection_f1", "detection_precision", "detection_recall", "tpr", "fpr"]
    out_path = run_dir / f"sweep_{sweep_tag}_defense_effectiveness.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Defense effectiveness CSV written to %s", out_path)
    print(f"\nDefense effectiveness: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join([
            str(row[sweep_param]),
            f"{row['detection_f1']:.4f}",
            f"{row['detection_precision']:.4f}",
            f"{row['detection_recall']:.4f}",
            f"{row['tpr']:.4f}",
            f"{row['fpr']:.4f}",
        ]))


def write_defense_effectiveness_filtered_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    fieldnames = [sweep_param, "detection_f1", "detection_precision", "detection_recall", "tpr", "fpr"]
    out_path = run_dir / f"sweep_{sweep_tag}_defense_effectiveness_filtered.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Filtered defense effectiveness CSV written to %s", out_path)
    print(f"\nDefense effectiveness (filtered): {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join([
            str(row[sweep_param]),
            f"{row['detection_f1']:.4f}",
            f"{row['detection_precision']:.4f}",
            f"{row['detection_recall']:.4f}",
            f"{row['tpr']:.4f}",
            f"{row['fpr']:.4f}",
        ]))


def write_attack_success_rate_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    fieldnames = [sweep_param, "attack_success_rate", "n_successful", "n_attacked_frames"]
    out_path = run_dir / f"sweep_{sweep_tag}_attack_success_rate.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Attack success rate CSV written to %s", out_path)
    print(f"\nAttack success rate: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join([
            str(row[sweep_param]),
            f"{row['attack_success_rate']:.4f}",
            str(row["n_successful"]),
            str(row["n_attacked_frames"]),
        ]))


def write_clustering_quality_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    fieldnames = [
        sweep_param,
        "spoofed_f1", "pred_f1", "precision", "spoofed_recall", "pred_recall",
    ]
    out_path = run_dir / f"sweep_{sweep_tag}_clustering_quality.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Clustering quality CSV written to %s", out_path)
    print(f"\nClustering quality: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join([
            str(row[sweep_param]),
            f"{row['spoofed_f1']:.4f}",
            f"{row['pred_f1']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['spoofed_recall']:.4f}",
            f"{row['pred_recall']:.4f}",
        ]))


def write_pacts_effectiveness_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    fieldnames = [sweep_param, "pacts_f1", "pacts_precision", "pacts_recall"]
    out_path = run_dir / f"sweep_{sweep_tag}_pacts_effectiveness.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("PACTS effectiveness CSV written to %s", out_path)
    print(f"\nPACTS effectiveness: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join([
            str(row[sweep_param]),
            f"{row['pacts_f1']:.4f}",
            f"{row['pacts_precision']:.4f}",
            f"{row['pacts_recall']:.4f}",
        ]))


def write_llm_attack_type_accuracy_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    fieldnames = [sweep_param, "n_tp_detected", "n_correct_type", "n_incorrect_type", "type_accuracy"]
    out_path = run_dir / f"sweep_{sweep_tag}_llm_attack_type_accuracy.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("LLM attack-type accuracy CSV written to %s", out_path)
    print(f"\nLLM attack-type accuracy: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join([
            str(row[sweep_param]),
            str(row["n_tp_detected"]),
            str(row["n_correct_type"]),
            str(row["n_incorrect_type"]),
            f"{row['type_accuracy']:.4f}",
        ]))


def write_llm_cost_metrics_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    _cost_fields = [
        f"{field}_{stat}"
        for field in ("input_tokens", "output_tokens", "thoughts_token_count")
        for stat in ("mean", "median", "std")
    ]
    fieldnames = [sweep_param] + _cost_fields + ["n_frames", "n_api_frames"]
    out_path = run_dir / f"sweep_{sweep_tag}_llm_cost_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("LLM cost metrics CSV written to %s", out_path)
    print(f"\nLLM cost metrics: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        num_vals = [
            f"{row.get(col, float('nan')):.2f}" if col not in ("n_frames", "n_api_frames")
            else str(row.get(col, ""))
            for col in fieldnames[1:]
        ]
        print(",".join([str(row[sweep_param])] + num_vals))


def write_timing_metrics_csv(
    run_dir: pathlib.Path,
    sweep_tag: str,
    sweep_param: str,
    rows: list[dict],
) -> None:
    timing_keys = sorted(
        k for k in rows[0] if k not in (sweep_param, "n_frames")
    )
    fieldnames = [sweep_param] + timing_keys + ["n_frames"]
    out_path = run_dir / f"sweep_{sweep_tag}_timing_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Timing metrics CSV written to %s", out_path)
    print(f"\nTiming metrics: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        num_vals = [
            f"{row.get(col, float('nan')):.4f}" if col != "n_frames"
            else str(row.get(col, ""))
            for col in fieldnames[1:]
        ]
        print(",".join([str(row[sweep_param])] + num_vals))
