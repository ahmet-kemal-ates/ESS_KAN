# ESS_KAN

Standalone KAN-based ESS model for one-step battery SoC prediction.

## What This Repo Does
- Loads RW battery MAT datasets (`trainRW*.mat`, `testRW*.mat`)
- Trains a KAN model on inputs `[Iin_A, Temp_C, SoC_prev]`
- Evaluates on unseen test data with `RMSE`, `MAE`, `R2`
- Exports a runtime artifact (`TorchScript + normalization metadata`)
- Provides a lightweight runtime wrapper for integration-style calls

## Project Structure
- `configs/base.yaml`: training and model config
- `data/dataset_generator_RWx.py`: generator for train/test RW datasets
- `src/data.py`: data loading + preprocessing
- `src/model.py`: efficient-KAN model definition
- `src/train.py`: training entrypoint
- `src/eval.py`: test evaluation
- `src/export.py`: runtime export + CPU benchmark
- `src/runtime_wrapper.py`: inference wrapper (`predict_next_soc`)
- `reports/rec_integration_spec.md`: REC integration mapping

## Dataset Provenance
The included train/test MAT files are derived from NASA PCoE battery aging data, Randomized Battery Usage (RW profile family):
- Source page: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Included generated files in `data/`:
- `trainRW9.mat` (28296 samples)
- `testRW9.mat` (20955 samples)
- `trainRW10.mat` (28382 samples)
- `testRW10.mat` (22258 samples)
- `trainRW11.mat` (27660 samples)
- `testRW11.mat` (25064 samples)
- `trainRW12.mat` (26704 samples)
- `testRW12.mat` (16224 samples)

MAT schema used by this project:
- Train keys: `tr_Time, tr_Iin, tr_SoC, tr_Vout, Temp, Cn, Ts`
- Test keys: `ts_Time, ts_Iin, ts_SoC, ts_Vout, Temp, Cn, Ts`

## How the Dataset Generator Works
Generator script: `data/dataset_generator_RWx.py`

It does the following:
1. Loads an RWx source MAT file (`m["data"].step`).
2. Detects random-walk blocks from step comments containing `random walk` or `pulsed load`.
3. Uses the first two RW blocks:
   - block 1 -> train
   - block 2 -> test
4. Selects the first `n` charge/discharge pulses (`type` in `C`/`D`) per block.
5. Resamples each pulse on a fixed grid (`Ts`, default `1.0 s`):
   - current: zero-order hold
   - voltage/temperature: linear interpolation
6. Computes SoC by coulomb counting (`Cn`, default `2.1 Ah`), clipped to `[0,1]`.
7. Saves ENNC-compatible MAT outputs (`trainRWx.mat`, `testRWx.mat`).

Example generation command:
```powershell
python data/dataset_generator_RWx.py --rw path\to\RW9.mat --pulses 100 --Cn 2.1 --Ts 1.0 --outdir data --yes
```

## Environment Setup (Windows, Conda)
```powershell
conda create -n ess_kan python=3.11 -y
conda activate ess_kan
pip install --upgrade pip
pip install -r requirements.txt
```

## Train
```powershell
python -m src.train --config configs/base.yaml
```

Output:
- `artifacts/run_YYYYMMDD_HHMMSS/model_best.pt`
- `artifacts/run_YYYYMMDD_HHMMSS/norm.npz`
- `artifacts/run_YYYYMMDD_HHMMSS/train_history.json`
- `artifacts/run_YYYYMMDD_HHMMSS/train_summary.json`

## Evaluate
```powershell
python -m src.eval --config configs/base.yaml --artifact_dir artifacts/run_YYYYMMDD_HHMMSS
```

Output:
- `artifacts/run_YYYYMMDD_HHMMSS/test_metrics.json`

## Export + Runtime Benchmark
```powershell
python -m src.export --config configs/base.yaml --artifact_dir artifacts/run_YYYYMMDD_HHMMSS
```

Output:
- `artifacts/run_YYYYMMDD_HHMMSS/export/model_ts.pt`
- `artifacts/run_YYYYMMDD_HHMMSS/export/norm.json`
- `artifacts/run_YYYYMMDD_HHMMSS/export/model_meta.json`
- `artifacts/run_YYYYMMDD_HHMMSS/runtime_benchmark.json`

## Runtime Wrapper Example
```powershell
python -c "from src.runtime_wrapper import ESSKANRuntime; rt=ESSKANRuntime.from_artifact_dir('artifacts/run_YYYYMMDD_HHMMSS'); print(rt.predict_next_soc(2.0,25.0,0.65))"
```

## Reproducibility Notes
- `requirements.txt` pins core package versions used in this project
- `efficient-kan` is pinned by Git commit
- model artifacts are timestamped under `artifacts/`

## Current Standalone Result (Latest Full Run)
- Run: `artifacts/run_20260306_010257`
- Test RMSE: `0.0026509`
- Test MAE: `0.0006519`
- Test R2: `0.9998291`
- CPU latency: `~0.405 ms/step`
