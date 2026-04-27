# FYP Experiment Pipeline

An adversarial attack and defense evaluation pipeline for LiDAR-based 3D object detectors used for autonomous vehicles. The project investigates the effectiveness of adversarial attacks against detectors used in autonomous vehicle perception, and evaluates various defenses that aim to detect the presense of an adversarial attack.

## Background

The pipeline evaluates two things:

1. **Attack effectiveness** — how much an attack degrades detector performance (measured via various metrics including average precision, precision-recall curves etc).
2. **Defense effectiveness** — how well a defense detects frames that have been attacked (measured via TPR, FPR, F1).

Detectors supported: 
- **PointPillars** (via OpenPCDet)
- **PointRCNN** (via OpenPCDet)

Dataset: 
- **KITTI**.

## Repository Structure

```
FYP-experiment-pipeline/
│
├── eval_pipeline/              # Core evaluation framework (main package)
│   ├── base.py                 # Abstract base classes: BaseAttack, BaseDetector, BaseDefense
│   ├── types.py                # Core dataclasses: Frame, Prediction, ObjectLabel, EvalResults, etc.
│   ├── pipeline.py             # EvalPipeline: orchestrates attack → detect → defend per frame
│   ├── config.py               # ExperimentConfig: Serialisable experiment specification
│   ├── runner.py               # CLI and programmatic entry points
│   ├── metrics.py              # KITTI-compliant AP, PR curves, recall-vs-IoU, defense metrics
│   ├── attacks/
│   │   └── ora.py              # Object Removal Attack (ORA) — two variants
│   ├── defenses/
│   │   └── void_region.py      # Void-Region Defense — shadow-based attack detector
│   ├── detectors/
│   │   ├── pointpillars.py     # PointPillars wrapper (OpenPCDet)
│   │   └── pointrcnn.py        # PointRCNN wrapper (OpenPCDet)
│   └── datasets/
│       ├── kitti.py            # KittiObjectDataset loader
│       └── kitti_utils.py      # KITTI calibration/label parsing utilities
│
├── scripts/
│   ├── run_ora_sweep.py        # Sweep ORA over attack budgets; exports AP, PR, recall curves
│   └── run_sweep.py            # Generalised parameter sweep (attack, defense, or detector params)
│
├── tools/
│   ├── visualise_metrics.py    # Interactive CLI: plot AP vs budget, PR curves, recall-vs-IoU
│   ├── visualise_defense.py    # 3D isometric and BEV visualisation of void-region defense
│   └── format_metrics.py       # Aggregate per-run JSON results into summary CSV
│
├── models/
│   └── openpcdet/              # Pretrained detector weights (.pth files)
│
├── results/                    # Timestamped experiment output directories (JSON + CSV)
│
├── draft_implementations/      # Reference implementations of attacks and defenses
│   ├── attacks/
│   └── defences/
│
├── devkit_object/              # KITTI devkit (C++/MATLAB) — reference for evaluation metrics
│
├── OpenPCDet/                  # OpenPCDet library clone (gitignored; must be installed locally)
│
├── attack_utils.py             # Utility for saving point clouds as KITTI .bin files
├── pointpillars_installation_guide.md     # Setup instructions for OpenPCDet integration
├── pixi.toml                   # Project manifest: dependencies and task shortcuts
└── pixi.lock                   # Locked dependency versions
```

### Gitignored directories

The following are excluded from version control and will not be present after cloning:

| Path | Reason |
|------|--------|
| `OpenPCDet/` | External library clone; install separately (see below) |
| `data/` | KITTI dataset; download separately |
| `.pixi/` | Pixi virtual environment |

## Pipeline Architecture

```
ExperimentConfig (YAML / CLI)
        │
        ▼
  EvalPipeline.run(dataset)
        │
        ├── For each frame:
        │     1. Detect clean points  (cached by frame ID)
        │     2. Apply attack         (optional; stochastic via attack_fraction)
        │     3. Detect objects on attacked points
        │     4. Run defense          (optional; receives rolling history window)
        │
        └── EvalResults
              ├── attack_effectiveness()  →  AP before vs. after attack
              └── defense_effectiveness() →  TPR / FPR / F1
```


### Metrics

- **AP** — R40-style Average Precision, per class and KITTI difficulty (Easy / Moderate / Hard)
- **PR curves** — Precision-Recall at varied confidence thresholds
- **Recall-vs-IoU** — Recall swept over IoU matching thresholds
- **Defense metrics** — True Positive Rate, False Positive Rate, Precision, Recall, F1

## Setup

### Prerequisites

- [Pixi](https://prefix.dev/tools/pixi) for environment management
- CUDA (see the [pointpillars installation guide](pointpillars_installation_guide.md) for more information on versions)

### Install

1. Install Python dependencies by running `pixi install`

2. Clone and install [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) into the project root (see the [pointpillars installation guide](pointpillars_installation_guide.md) for full instructions)

3. Place pretrained weights in models/openpcdet/ (links for download can be found in the readme at [OpenPCDet](https://github.com/open-mmlab/OpenPCDet))

4. Download KITTI Object Detection dataset and update `KITTI_ROOT` variable in scripts


## Usage

### Premade command shortcuts
A variety of commands can be found in the [pixi.toml](pixi.toml) file under `[tasks]`. To run them, just run `pixi run task-name`. This is the recommended way to run the pipeline as it makes going back and forth between different cmd args very simple.
```bash
pixi run s-ora           # Short ORA sweep (attack only)
pixi run s-ora-vr        # Short ORA sweep with void-region defense
pixi run vm              # Visualise metrics
pixi run vd              # Visualise defense
pixi run fm              # Format/aggregate metrics to CSV
```

### Manually run ORA budget sweep

```bash
python scripts/run_ora_sweep.py \
    --split val --num-frames 50 \
    --budgets 0 10 40 100 200 \
    --classes Car Pedestrian Cyclist \
    --difficulties Easy Moderate Hard \
    --metric-types ap pr recall_iou
```

### Manually run a defense parameter sweep

```bash
python scripts/run_sweep.py \
    --attack ora --defense void_region --detector pointrcnn \
    --attack-fraction 0.5 \
    --sweep-target defense --sweep-param grid_stride \
    --sweep-values 0.3 0.4 0.5 0.6
```

### Single experiment via CLI

```bash
python -m eval_pipeline.runner \
    --kitti-root data/datasets/KITTI \
    --attack ora --budget 200 \
    --detector pointrcnn \
    --frames 000125 000070 \
    --experiment-name test_ora_200
```

### Visualise results

```bash
python tools/visualise_metrics.py --results-dir results
python tools/visualise_defense.py --isometric
```


## Python API

```python
from eval_pipeline import EvalPipeline, KittiObjectDataset
from eval_pipeline.attacks import ORAAttack
from eval_pipeline.defenses import VoidRegionDefense
from eval_pipeline.detectors import PointRCNNDetector

dataset = KittiObjectDataset("data/datasets/KITTI", frame_ids=["000125", "000070"])
pipeline = EvalPipeline(
    dataset,
    attack=ORAAttack(budget=200),
    detector=PointRCNNDetector(),
    defense=VoidRegionDefense(),
)
results = pipeline.run()
print(results.attack_effectiveness())
print(results.defense_effectiveness())
```
