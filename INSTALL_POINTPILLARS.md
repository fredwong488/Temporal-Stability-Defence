# Installing PointPillars via OpenPCDet

This guide covers installing [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) to run PointPillars,
and integrating it with the `eval_pipeline` in this repo.

---

## Requirements

| Component | Version |
|-----------|---------|
| OS        | Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04) |
| Python    | 3.6+ |
| PyTorch   | 1.1 or higher (tested on PyTorch 1.1, 1.3, 1.5~1.10) |
| CUDA      | 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+) |
| spconv    | v1.0 (commit 8da6f96) or v1.2 or v2.x |

> **spconv v2.x requires PyTorch >= 1.5.0.** Supported CUDA versions:
>
> | CUDA | Install command |
> |------|----------------|
> | 10.2 | `pip install spconv-cu102` |
> | 11.3 | `pip install spconv-cu113` |
> | 11.4 | `pip install spconv-cu114` |
> | 11.6 | `pip install spconv-cu116` |
> | 11.7 | `pip install spconv-cu117` |
> | 11.8 | `pip install spconv-cu118` |
> | **12.0** | **`pip install spconv-cu120`** ← recommended |

> **CUDA alignment is critical.** Your system CUDA toolkit (`nvcc --version`), PyTorch's
> bundled CUDA (`python -c "import torch; print(torch.version.cuda)"`), and your
> `spconv-cuXXX` pip package must all point at the **same** CUDA version.

---

## Step 1 — Install the pixi Environment

This project uses [pixi](https://prefix.dev/) to manage dependencies. PyTorch (CUDA 12.0)
and `spconv-cu120` are declared in `pixi.toml` and will be installed automatically.

```bash
pixi install
pixi shell
```

Verify:
```bash
python --version
# Expected: Python 3.11.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import spconv; print('spconv', spconv.__version__)"
```

---

## Step 2 — Clone and Install OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
```

Remove the `torch>=1.1` line from `requirements.txt` to prevent pip overwriting the
PyTorch install above:
```bash
sed -i '/^torch/d' requirements.txt
```

Build `pcdet`:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

> **Note:** Always use `python -m pip` rather than `pip` directly. Inside a pixi shell,
> bare `pip` may resolve to the system pip, which will refuse to install into the
> pixi-managed environment. `python -m pip` uses the pip belonging to the active
> Python interpreter and avoids this.
>
> `--no-build-isolation` is required because OpenPCDet's `setup.py` imports `torch`
> at build time (to detect CUDA version for compiling extensions). Without this flag,
> pip creates an isolated build environment in `/tmp` that does not have access to the
> pixi-managed torch, causing a `ModuleNotFoundError: No module named 'torch'` error.

or alternatively if for any reason the dependencies are not installed in your pixi environment yet:
```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

Verify:
```bash
python -c "import pcdet; print('pcdet OK')"
```

---

## Step 3 — Download Pre-trained PointPillars Weights

The KITTI-trained checkpoint is available from the OpenPCDet model zoo:

- **Config:** `tools/cfgs/kitti_models/pointpillar.yaml`
- **Checkpoint:** [pointpillar_7728.pth (18 MB) — Google Drive](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view)

Download and place it somewhere accessible, e.g.:
```
OpenPCDet/checkpoints/pointpillar_7728.pth
```

---

## Step 4 — (Optional) Verify with the Built-in Demo

Run OpenPCDet's demo on a single KITTI `.bin` file to confirm the model loads:

```bash
cd OpenPCDet
python demo.py \
    --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
    --ckpt checkpoints/pointpillar_7728.pth \
    --data_path /path/to/kitti/testing/velodyne/000000.bin
```

---

## Step 5 — Wire into eval_pipeline

In `eval_pipeline/detectors/pointpillars.py`, implement `_load_model()` and
`_run_inference()`, then point your experiment config at the weights:

```yaml
# experiment.yaml
detector_type: pointpillars
detector_params:
  config_path: /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
  checkpoint_path: /path/to/OpenPCDet/checkpoints/pointpillar_7728.pth
  score_threshold: 0.3
  device: cuda:0
```

---

## Known Issues and possible fixes

### numba / llvmlite conflict
If `python -m pip install -r requirements.txt` fails with numba errors:
```bash
python -m pip uninstall numba llvmlite -y
python -m pip install llvmlite==0.39.1 numba==0.56.4 --force-reinstall
```

### GCC version too new for CUDA 12.0
CUDA 12.0's `nvcc` supports up to GCC 12. If your system GCC is newer you will see:
```
error: #error -- unsupported GNU version! gcc versions later than 12 are not supported!
```
Fix by pointing the compiler at GCC 12 before building:
```bash
ls /usr/bin/gcc-*          # check which versions are installed
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
python -m pip install -e . --no-deps --no-build-isolation
```
These exports only apply to the current shell session — you will need to re-set them
if you open a new terminal.

### CUDA environment variables not set
Before `python setup.py develop`, ensure:
```bash
export CUDA_HOME=/vol/cuda/12.0.0
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Re-install after any env change
The OpenPCDet docs explicitly require re-running this after any environment change:
```bash
cd OpenPCDet && python -m pip install -e . --no-deps --no-build-isolation
```

### spconv v1 vs v2 import clash
In spconv v2, the import changed to `import spconv.pytorch as spconv`. OpenPCDet
handles this internally, but if you see `AttributeError: module 'spconv' has no
attribute '__version__'`, you have mixed v1/v2 installed — uninstall all spconv
packages and reinstall only `spconv-cu120`.
