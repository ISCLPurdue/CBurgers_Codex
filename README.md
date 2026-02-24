# CBurgers_Codex

This mini-application demonstrates a C++ and Python scientific machine learning workflow:

1. C++ solves a 1D viscous Burgers equation and streams snapshots in situ.
2. Python performs SVD-based modal compression.
3. An LSTM model is trained on modal coefficients and used for forecasting.
4. Result figures are generated directly from the workflow.

For project context, see the original paper link in the repo history: [10.1063/5.0019884](https://doi.org/10.1063/5.0019884).

## Session Deliverables

- Browser-renderable dashboard: [`docs/dashboard.html`](docs/dashboard.html)
- Browser-renderable report (PDF): [`docs/changes_summary.pdf`](docs/changes_summary.pdf)
- GitHub Pages entry point (after enabling Pages): `https://isclpurdue.github.io/CBurgers_Codex/`

## Current Build and Run Instructions

These steps match the current source code and dependency setup.

### 1. Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install numpy scipy matplotlib jupyter cmake tensorflow scikit-learn
```

### 3. Configure and build C++

```bash
.venv/bin/cmake -S . -B build_local -DPython3_EXECUTABLE=.venv/bin/python
.venv/bin/cmake --build build_local -j
```

### 4. Run from `build/` so Python modules are found

The runtime imports `python_module.py` from the current working directory (`build/`), and embedded Python needs access to venv packages.

```bash
cd build
PYTHONPATH="$(../.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
MPLCONFIGDIR=../.mplconfig \
../build_local/app
```

## Output Artifacts (Current Paths)

Generated outputs are in `build/`:

- `build/Field_evolution.png`
- `build/SVD_Eigenvectors.png`
- `build/Mode_0_prediction.png`
- `build/Mode_1_prediction.png`
- `build/Mode_2_prediction.png`
- `build/eigenvectors.npy`
- `build/checkpoints/my_checkpoint.weights.h5`

## Result Preview

### Field evolution
![Fields](build/Field_evolution.png "Fields")

### Modal decomposition
![Modes](build/SVD_Eigenvectors.png "Modes")

### Forecasting modal evolution
![Forecasting Mode 0](build/Mode_0_prediction.png "Mode 0 prediction")
![Forecasting Mode 1](build/Mode_1_prediction.png "Mode 1 prediction")
![Forecasting Mode 2](build/Mode_2_prediction.png "Mode 2 prediction")
