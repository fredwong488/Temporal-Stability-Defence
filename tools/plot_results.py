"""
tools/plot_results.py
---------------------
Interactive Plotly Dash explorer for *_frames.jsonl result files.

Run:
    pixi run python tools/plot_results.py [results/dir1 results/dir2 ...]
    pixi run pr                          # task alias, scans all of results/

If no directories are given, the script scans the entire results/ folder.

JSONL files are loaded lazily — only when first selected in the checklist.
Each file's sibling .json (experiment config + defense_effectiveness) is read
at startup for the "Run info" panel without loading the heavy JSONL data.

Two DataFrames are assembled from whichever files are currently selected:

    df_frame   — one row per frame
        file, run, frame_id, sequence_id, frame_index_in_scene, scene_length,
        is_attacked, is_attack_detected, confidence,
        n_clusters_tested, n_clusters_flagged,
        centroid_threshold, point_threshold, defense_reason

    df_cluster — one row per (frame × cluster)
        all df_frame columns plus:
        n_points_cur, n_frames_associated,
        sigma_centroid, sigma_point,
        flagged_centroid, flagged_point, flagged,
        centroid_x, centroid_y, centroid_z
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).parent.parent / "results"

FRAME_NUMERIC = [
    "frame_index_in_scene", "scene_length", "confidence",
    "n_clusters_tested", "n_clusters_flagged",
    "centroid_threshold", "point_threshold",
]
FRAME_CATEGORICAL = ["file", "run", "is_attacked", "is_attack_detected", "defense_reason"]

CLUSTER_NUMERIC = FRAME_NUMERIC + [
    "n_points_cur", "n_frames_associated",
    "sigma_centroid", "sigma_point",
    "centroid_x", "centroid_y", "centroid_z",
]
CLUSTER_CATEGORICAL = FRAME_CATEGORICAL + ["flagged_centroid", "flagged_point", "flagged", "is_spoofed_cluster"]

FILTER_COLS = [
    "is_attacked", "is_attack_detected", "flagged",
    "flagged_centroid", "flagged_point", "is_spoofed_cluster",
    "file", "run", "defense_reason",
]

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_jsonl_files(dirs: list[Path]) -> list[Path]:
    files = []
    for d in dirs:
        files.extend(sorted(d.glob("*_frames.jsonl")))
    return files


def load_run_config(jsonl_path: Path) -> dict:
    """Read the sibling .json summary file for config + effectiveness stats."""
    # e.g. ora_point_threshold_0.2_frames.jsonl → ora_point_threshold_0.2.json
    stem = jsonl_path.stem  # "…_frames"
    if stem.endswith("_frames"):
        stem = stem[: -len("_frames")]
    sibling = jsonl_path.parent / f"{stem}.json"
    if not sibling.exists():
        return {}
    try:
        with sibling.open() as fh:
            return json.load(fh)
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Lazy data loading
# ---------------------------------------------------------------------------

# Server-side cache: path_str → (df_frame, df_cluster)
_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def _load_jsonl(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = path.parent.name
    fname = path.stem

    rows_frame: list[dict] = []
    rows_cluster: list[dict] = []

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            dr = d.get("defense_result") or {}
            meta = dr.get("metadata") or {}

            frame_row: dict = {
                "file": fname,
                "run": run,
                "frame_id": d.get("frame_id"),
                "sequence_id": d.get("sequence_id"),
                "frame_index_in_scene": d.get("frame_index_in_scene"),
                "scene_length": d.get("scene_length"),
                "is_attacked": d.get("is_attacked"),
                "is_attack_detected": dr.get("is_attack_detected"),
                "confidence": dr.get("confidence"),
                "n_clusters_tested": meta.get("n_clusters_tested"),
                "n_clusters_flagged": meta.get("n_clusters_flagged"),
                "centroid_threshold": meta.get("centroid_threshold"),
                "point_threshold": meta.get("point_threshold"),
                "defense_reason": meta.get("reason"),
            }
            rows_frame.append(frame_row)

            # Collect spoofed-cluster centroids from attack_metadata (ORA only).
            # Each entry has a "reinjected_centroid" [x, y, z] when n_removed > 0.
            spoofed_centroids: list[tuple[float, float, float]] = []
            for obj in (d.get("attack_metadata") or {}).get("removed_per_obj") or []:
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
                    is_spoofed = min_dist < 1.5

                rows_cluster.append({
                    **frame_row,
                    "n_points_cur": cd.get("n_points_cur"),
                    "n_frames_associated": cd.get("n_frames_associated"),
                    "sigma_centroid": cd.get("sigma_centroid"),
                    "sigma_point": cd.get("sigma_point"),
                    "flagged_centroid": cd.get("flagged_centroid"),
                    "flagged_point": cd.get("flagged_point"),
                    "flagged": cd.get("flagged"),
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "centroid_z": cz,
                    "is_spoofed_cluster": is_spoofed,
                })

    return pd.DataFrame(rows_frame), pd.DataFrame(rows_cluster)


def get_dataframes(selected_paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return combined (df_frame, df_cluster) for the selected files, loading lazily."""
    frames_f, frames_c = [], []
    for path_str in selected_paths:
        if path_str not in _cache:
            p = Path(path_str)
            print(f"  Loading {p.parent.name}/{p.name} …", flush=True)
            _cache[path_str] = _load_jsonl(p)
        df_f, df_c = _cache[path_str]
        frames_f.append(df_f)
        frames_c.append(df_c)

    df_frame = pd.concat(frames_f, ignore_index=True) if frames_f else pd.DataFrame()
    df_cluster = pd.concat(frames_c, ignore_index=True) if frames_c else pd.DataFrame()
    return df_frame, df_cluster

# ---------------------------------------------------------------------------
# Run-info panel
# ---------------------------------------------------------------------------

def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def build_run_info_panel(cfg: dict) -> html.Div:
    if not cfg:
        return html.Div("No config file found.", style={"fontSize": "0.78rem", "color": "#999"})

    config = cfg.get("config", {})
    eff = cfg.get("defense_effectiveness", {})

    def row(label, value):
        return html.Tr([
            html.Td(label, style={"color": "#666", "paddingRight": "8px", "whiteSpace": "nowrap", "fontSize": "0.75rem"}),
            html.Td(_fmt_val(value), style={"fontWeight": "500", "fontSize": "0.75rem", "wordBreak": "break-all"}),
        ])

    attack_params = config.get("attack_params", {})
    defense_params = config.get("defense_params", {})
    dataset_params = config.get("dataset_params", {})

    basic_rows = [
        row("experiment", cfg.get("experiment_name", "—")),
        row("dataset", f"{config.get('dataset_type', '—')}  {dataset_params.get('split', '')}"),
        row("detector", config.get("detector_type", "—")),
        row("attack", config.get("attack_type", "—")),
        row("attack_fraction", config.get("attack_fraction", "—")),
        row("defense", config.get("defense_type", "—")),
        row("num_frames", cfg.get("num_frames", "—")),
    ]
    for k, v in defense_params.items():
        basic_rows.append(row(f"  {k}", v))

    eff_rows = []
    if eff:
        for k in ("tp", "fp", "tn", "fn", "tpr", "fpr"):
            if k in eff:
                eff_rows.append(row(k, eff[k]))

    children = [
        html.Table(basic_rows, style={"borderCollapse": "collapse", "width": "100%"}),
    ]
    if eff_rows:
        children += [
            html.Hr(style={"margin": "6px 0"}),
            html.Table(eff_rows, style={"borderCollapse": "collapse", "width": "100%"}),
        ]

    return html.Div(children, style={"backgroundColor": "#f0f4ff", "padding": "8px", "borderRadius": "4px"})

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_layout(file_options: list[dict]) -> html.Div:
    filter_rows = []
    for col in FILTER_COLS:
        filter_rows.append(html.Div([
            html.Label(col, style={"fontSize": "0.78rem", "marginBottom": "2px"}),
            dcc.Dropdown(
                id=f"filter-{col}",
                options=[],
                multi=True,
                placeholder=f"All",
                style={"fontSize": "0.78rem"},
            ),
        ], style={"marginBottom": "6px"}))

    sidebar = html.Div([
        # --- File selection ---
        html.H4("Files", style={"margin": "0 0 4px 0"}),
        html.Div(
            "Files are loaded on first selection.",
            style={"fontSize": "0.72rem", "color": "#888", "marginBottom": "6px"},
        ),
        dcc.Checklist(
            id="file-checklist",
            options=file_options,
            value=[],
            labelStyle={"display": "block", "fontSize": "0.73rem", "overflowWrap": "anywhere"},
        ),
        html.Hr(),

        # --- Run info ---
        html.H4("Run info", style={"margin": "0 0 4px 0"}),
        dcc.Dropdown(
            id="inspect-file",
            options=file_options,
            placeholder="Inspect a file's config…",
            clearable=True,
            style={"fontSize": "0.78rem", "marginBottom": "6px"},
        ),
        html.Div(id="run-info-panel"),
        html.Hr(),

        # --- Plot controls ---
        html.H4("Granularity", style={"margin": "0 0 4px 0"}),
        dcc.RadioItems(
            id="granularity",
            options=[
                {"label": " Per frame", "value": "frame"},
                {"label": " Per cluster", "value": "cluster"},
            ],
            value="cluster",
            labelStyle={"display": "block"},
        ),
        html.Hr(),
        html.H4("Plot type", style={"margin": "0 0 4px 0"}),
        dcc.RadioItems(
            id="plot-type",
            options=[
                {"label": " Histogram", "value": "histogram"},
                {"label": " Box", "value": "box"},
                {"label": " Scatter", "value": "scatter"},
                {"label": " ECDF", "value": "ecdf"},
            ],
            value="histogram",
            labelStyle={"display": "block"},
        ),
        html.Hr(),
        html.H4("X variable", style={"margin": "0 0 4px 0"}),
        dcc.Dropdown(id="x-col", clearable=False),
        html.Div(id="y-col-div", children=[
            html.H4("Y variable", style={"margin": "8px 0 4px 0"}),
            dcc.Dropdown(id="y-col"),
        ], style={"display": "none"}),
        html.H4("Colour by", style={"margin": "8px 0 4px 0"}),
        dcc.Dropdown(id="color-col"),
        html.Hr(),
        html.H4("Filters", style={"margin": "0 0 4px 0"}),
        *filter_rows,
    ], style={
        "width": "270px",
        "minWidth": "270px",
        "padding": "12px",
        "overflowY": "auto",
        "height": "100vh",
        "boxSizing": "border-box",
        "borderRight": "1px solid #ddd",
        "backgroundColor": "#fafafa",
    })

    main = html.Div([
        html.Div(id="load-status", style={
            "fontSize": "0.82rem", "color": "#1565c0",
            "marginBottom": "4px", "minHeight": "1.2em",
        }),
        html.Div(id="stats-bar", style={
            "fontSize": "0.82rem", "color": "#555",
            "marginBottom": "8px",
        }),
        dcc.Graph(id="main-graph", style={"height": "88vh"}),
    ], style={"flex": 1, "padding": "12px", "overflowY": "auto"})

    return html.Div([sidebar, main], style={"display": "flex", "fontFamily": "sans-serif"})

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        search_dirs = [Path(a) for a in sys.argv[1:]]
    else:
        search_dirs = sorted(RESULTS_ROOT.iterdir()) if RESULTS_ROOT.exists() else []

    all_files = find_jsonl_files(search_dirs)
    if not all_files:
        print("No *_frames.jsonl files found. Pass result directories as arguments.")
        sys.exit(1)

    print(f"Found {len(all_files)} JSONL file(s) (not yet loaded).")

    # Pre-load sibling .json configs — lightweight, instant
    configs: dict[str, dict] = {str(p): load_run_config(p) for p in all_files}

    file_options = []
    seen: set[str] = set()
    for p in all_files:
        val = str(p)
        if val not in seen:
            seen.add(val)
            file_options.append({"label": f"{p.parent.name}/{p.stem}", "value": val})

    app = Dash(__name__, title="Results Explorer")
    app.layout = build_layout(file_options)

    # ------------------------------------------------------------------
    # Run-info panel
    # ------------------------------------------------------------------

    @app.callback(
        Output("run-info-panel", "children"),
        Input("inspect-file", "value"),
    )
    def update_run_info(path_str):
        if not path_str:
            return html.Div(
                "Select a file above to inspect its config.",
                style={"fontSize": "0.75rem", "color": "#aaa"},
            )
        cfg = configs.get(path_str, {})
        return build_run_info_panel(cfg)

    # ------------------------------------------------------------------
    # Column dropdowns (depend on granularity + plot type only)
    # ------------------------------------------------------------------

    @app.callback(
        Output("x-col", "options"),
        Output("x-col", "value"),
        Output("y-col", "options"),
        Output("y-col", "value"),
        Output("color-col", "options"),
        Output("color-col", "value"),
        Output("y-col-div", "style"),
        Input("granularity", "value"),
        Input("plot-type", "value"),
        State("x-col", "value"),
        State("y-col", "value"),
        State("color-col", "value"),
    )
    def update_col_dropdowns(granularity, plot_type, cur_x, cur_y, cur_color):
        numeric = CLUSTER_NUMERIC if granularity == "cluster" else FRAME_NUMERIC
        categorical = CLUSTER_CATEGORICAL if granularity == "cluster" else FRAME_CATEGORICAL
        all_cols = numeric + categorical

        num_opts = [{"label": c, "value": c} for c in numeric]
        all_opts = [{"label": c, "value": c} for c in all_cols]

        needs_y = plot_type == "scatter"
        y_style = {"display": "block"} if needs_y else {"display": "none"}

        default_x = cur_x if cur_x in all_cols else (numeric[0] if numeric else None)
        default_y = cur_y if (needs_y and cur_y in all_cols) else (numeric[1] if needs_y and len(numeric) > 1 else None)
        default_color = cur_color if cur_color in all_cols else None

        return num_opts, default_x, num_opts, default_y, all_opts, default_color, y_style

    # ------------------------------------------------------------------
    # Filter dropdowns (reset values when selection changes)
    # ------------------------------------------------------------------

    filter_outputs = (
        [Output(f"filter-{col}", "options") for col in FILTER_COLS]
        + [Output(f"filter-{col}", "value") for col in FILTER_COLS]
    )

    @app.callback(
        filter_outputs,
        Input("file-checklist", "value"),
        Input("granularity", "value"),
    )
    def update_filter_options(selected_files, granularity):
        if not selected_files:
            return [[]] * len(FILTER_COLS) + [None] * len(FILTER_COLS)

        df_frame, df_cluster = get_dataframes(selected_files)
        df = df_cluster if granularity == "cluster" else df_frame

        opts = []
        for col in FILTER_COLS:
            if col in df.columns:
                uniq = sorted(df[col].dropna().unique(), key=str)
                opts.append([{"label": str(v), "value": str(v)} for v in uniq])
            else:
                opts.append([])
        return opts + [None] * len(FILTER_COLS)

    # ------------------------------------------------------------------
    # Main graph
    # ------------------------------------------------------------------

    @app.callback(
        Output("main-graph", "figure"),
        Output("stats-bar", "children"),
        Output("load-status", "children"),
        Input("file-checklist", "value"),
        Input("granularity", "value"),
        Input("plot-type", "value"),
        Input("x-col", "value"),
        Input("y-col", "value"),
        Input("color-col", "value"),
        *[Input(f"filter-{col}", "value") for col in FILTER_COLS],
    )
    def update_graph(selected_files, granularity, plot_type, x_col, y_col, color_col, *filter_vals):
        if not selected_files:
            return go.Figure(), "No files selected.", ""

        cached_before = set(_cache.keys())
        df_frame, df_cluster = get_dataframes(selected_files)
        newly_loaded = [Path(p).stem for p in set(_cache.keys()) - cached_before]
        load_msg = f"Loaded: {', '.join(newly_loaded)}" if newly_loaded else ""

        df = (df_cluster if granularity == "cluster" else df_frame).copy()

        # Apply filters
        for col, vals in zip(FILTER_COLS, filter_vals):
            if vals and col in df.columns:
                df = df[df[col].astype(str).isin(vals)]

        if df.empty or not x_col:
            return go.Figure(), "No data after filters.", load_msg

        hover = ["frame_id", "file", "run"]
        if granularity == "cluster":
            hover += ["sigma_centroid", "sigma_point", "n_points_cur", "n_frames_associated", "flagged"]
        hover = [c for c in hover if c in df.columns]

        color = color_col if color_col and color_col in df.columns else None
        if color is not None:
            df[color] = df[color].astype(str)

        try:
            if plot_type == "histogram":
                fig = px.histogram(
                    df, x=x_col, color=color,
                    barmode="overlay", opacity=0.7,
                    marginal="rug",
                    labels={x_col: x_col},
                )
            elif plot_type == "box":
                fig = px.box(
                    df, x=color, y=x_col,
                    points="outliers",
                    labels={x_col: x_col},
                )
            elif plot_type == "scatter":
                if not y_col or y_col not in df.columns:
                    return go.Figure(), "Select a Y variable for scatter.", load_msg
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=color,
                    opacity=0.5,
                    hover_data=hover,
                    labels={x_col: x_col, y_col: y_col},
                )
            elif plot_type == "ecdf":
                fig = px.ecdf(
                    df, x=x_col, color=color,
                    labels={x_col: x_col},
                )
            else:
                return go.Figure(), "Unknown plot type.", load_msg
        except Exception as exc:
            return go.Figure(), f"Plot error: {exc}", load_msg

        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        n = len(df)
        stats = f"{n:,} rows | {x_col}"
        if x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
            col_data = df[x_col].dropna()
            if len(col_data):
                stats += (
                    f"  mean={col_data.mean():.4g}"
                    f"  std={col_data.std():.4g}"
                    f"  min={col_data.min():.4g}"
                    f"  max={col_data.max():.4g}"
                )

        return fig, stats, load_msg

    print("Starting Dash app at http://127.0.0.1:8050/")
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
