# Submarine Inlet BC & Drag Estimator — Detailed Work Doc

**Repo:** https://github.com/appars/submarine-poc  
**Live App:** https://submarine-poc-aw4sk3aafktkv9mmer4gkb.streamlit.app/

---

## 1) What this app does

This app builds **fast surrogate models** that map high-level *submarine hull design parameters* → to
- **Inlet boundary conditions (BCs)**: free-stream speed \(U_\text{mag}\), turbulence intensity **TI**, integral length scale **Lt** (intended for k–ω SST / RANS setups).
- **Total drag coefficient \(C_d\)**.

It lets you:
1. **Generate synthetic training data** (or use OpenFOAM/Fluidyn data when available).
2. **Train surrogates** (XGBoost / LightGBM / sklearn pipelines) and save `models/surrogates.joblib`.
3. **Predict** BCs & \(C_d\) for a new design and **explain results** (human-readable “medical-check” style report).
4. *(Optional module — can be enabled later)* **Consume & Benchmark** OpenFOAM logs/VTK to quantify warm-start speedups and visualize fields.

No paid APIs are required. The LLM “explain” box is optional and **disabled by default** (can run locally via **Ollama** if you want).

---

## 2) Scope & assumptions

- Targets: \(U_\text{mag}\), **TI**, **Lt**, \(C_d\).
- Models are **regressors** trained on **synthetic CFD-like samples** unless you provide real CFD/Fluidyn datasets.
- Intended as a **POC** to reduce simulation time by providing *good initial BCs* and a *quick \(C_d\) estimate* prior to full RANS runs.
- When real data becomes available, simply drop CSVs into `data/` and retrain — pipelines remain the same.

---

## 3) How the pieces fit

```
Design params ──▶ Data generator (synthetic or real CSV)
                   │
                   ├──▶ Train pipelines (sklearn + XGB/LGBM) ──▶ surrogates.joblib
                   │
                   └──▶ Metrics report (MAE/MAPE/R², curves)
                                         │
Streamlit app ────────────────────────────┼────────────────▶ Prediction UI + “Explain my BCs” (rule-based)
(optional) OpenFOAM case outputs (logs/VTK) └──▶ Consume & Benchmark page (speedup, residuals, field viz)
```

---

## 4) Features / inputs

The exact feature set is configurable; the default synthetic generator expects columns like:
- **L_over_D** (fineness ratio), **nose_fineness**, **tail_cone_deg**, **AoA_deg**
- **Re** (Reynolds number), **Mach** (if transonic effects considered), **roughness** (nondimensional), **appendage_factor** (0..1)  
*(Your Streamlit sidebar shows the sliders/text inputs that map to these.)*

> **Tip:** When you plug in real design parameters, use consistent units and ranges. The training generator clamps to safe ranges and records them to compute medical-style normal bands.

---

## 5) Targets & BC conversions (for RANS k–ω SST)

The app predicts **TI** and **Lt** directly, but you may need **k** and **ω** to write OpenFOAM fields:

- \( k = \tfrac{3}{2}\,(U_\text{mag}\,TI)^2 \)
- \( \omega = \dfrac{\sqrt{k}}{C_\mu^{1/4}\,L_t} \)  with \(C_\mu = 0.09\)

These are included in the UI helper and docs so you can translate BCs quickly.

---

## 6) Training pipeline

### 6.1 Data sources
- **Synthetic generator** (default) — quickly creates physically plausible tuples \((X, y)\) following smooth rules + noise.
- **Real CFD (Fluidyn/OpenFOAM)** — drop your CSV in `data/real_*.csv` with the same schema (features + targets).

### 6.2 Models
- **BC regressors**: GradientBoostingRegressor / LightGBM / XGBoost (configurable).  
- **Drag coefficient \(C_d\)**: XGBoostRegressor (default, robust on tabular with nonlinearities).
- **Preprocessing**: `ColumnTransformer` + `StandardScaler` for numerics (and any one-hot for categoricals if present).  
- **Persistence**: `joblib` bundle at `models/surrogates.joblib` storing a dict:
  - `BC_U_mag`, `BC_TI`, `BC_Lt`, `Cd`, plus training scalers and feature lists.

### 6.3 Metrics
The training script prints and logs:
- **MAE**, **MAPE**, **RMSE**, **R²**
- **Learning curves** (optional)
- **Calibration plots** (optional)

Command examples:
```bash
# fresh venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train with 800 synthetic samples (fast)
python train_surrogates.py

# train with 3000 samples (better generalization)
python train_surrogates.py --n 3000 --seed 42
# outputs: models/surrogates.joblib  and  data/synthetic_*.csv
```

> **Mac ARM tip:** If you enable XGBoost and see `libomp.dylib` errors, run:  
> `brew install libomp` then reinstall `xgboost` inside the venv.

---

## 7) App surfaces (Streamlit)

### 7.1 Home / Predict
- Enter design parameters
- App loads `models/surrogates.joblib` (if present) and predicts:
  - \(U_\text{mag}\), **TI**, **Lt**, \(C_d\)
- Provides **copy-paste** snippets (k/ω formulas and example OpenFOAM dictionary inserts).

### 7.2 “Explain my BCs” (doctor’s report style)
- **Rule-based explanation** (no internet, no API):  
  Compares each prediction against **normal bands** computed from training quantiles.  
  Example:
  - **TI**: _Normal 0.5–5%_; **Pred** 2.7% – *OK*  
  - **Lt**: _Normal 0.05–0.5 D_; **Pred** 0.62 D – *Slightly high → consider shorter inlet development or coarser grid near inlet.*  
- Optional **LLM explanation** via **Ollama** (`localhost:11434`), disabled by default.

### 7.3 Consume & Benchmark (optional module)
- **Disabled by default** in your repo.
- When enabled, it lets you upload two case ZIPs (baseline vs warm-start) and:
  - Parse `log.simpleFoam` for `ExecutionTime` and residual timelines
  - Plot residuals and compute **speedup**
  - Optional `forceCoeffs.csv` overlay
  - Optional VTK slice viewer (`meshio` + `plotly`)

To re-enable, restore `pages/2_Consume_and_Benchmark.py` and add `meshio`, `plotly` to `requirements.txt`.

---

## 8) Local run & tests

```bash
# 1) set up
git clone https://github.com/appars/submarine-poc
cd submarine-poc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) (optional) generate larger synthetic dataset & train
python train_surrogates.py --n 3000

# 3) run the app
streamlit run app.py
# open http://localhost:8501
```

**Files of interest**
- `train_surrogates.py` — generates data, trains models, prints metrics
- `app.py` — Streamlit UI for prediction + explanation
- `models/surrogates.joblib` — saved pipelines
- `data/synthetic_*.csv` — synthetic training data snapshots

---

## 9) Deployment (Streamlit Community Cloud)

1. Connect your GitHub repo (`main` branch, entrypoint `app.py`).  
2. No secrets required (LLM turned off).  
3. If you later enable the optional page: add `meshio` and `plotly` to `requirements.txt`.  
4. Use Python 3.10–3.12 runtime for fastest cold starts.  
5. When you retrain locally, **commit the new `models/surrogates.joblib`** so the cloud app uses it.

---

## 10) Integration with CFD (OpenFOAM quick notes)

- From predicted \(U_\text{mag}\), TI, Lt compute \(k, \omega\) (formulas above).
- Inlet boundary examples (k–ω SST):
  ```
  0/U:
    inlet { type fixedValue; value uniform (U_mag 0 0); }

  0/k:
    inlet { type fixedValue; value uniform k_from_TI; }

  0/omega:
    inlet { type fixedValue; value uniform omega_from_k_Lt; }
  ```
- For **warm-start** demos, initialize `internalField` to inlet values and run a shorter solve with tighter `residualControl`, then compare with a baseline cold start.

---

## 11) Metrics to show in demos

- **Regression**: MAE, MAPE, RMSE, R² for each target (BC_U_mag, TI, Lt, Cd).  
- **Cost efficiency**: time-to-threshold vs baseline (if you enable the Benchmark page).  
- **Latency** (surrogate prediction): usually <1 ms per design on laptop.  
- **Calibration** (optional): predicted vs true scatter, ideal y=x overlay.

---

## 12) Troubleshooting (Mac)

- **XGBoost “libxgboost.dylib / libomp.dylib”**:  
  `brew install libomp` → reinstall `xgboost` in your venv.

- **Ollama not found** (only if you enable LLM):  
  Install Ollama app, run `ollama serve`, then `ollama pull llama3.1:8b`.  
  In the app, set `LLM_MODE=ollama` (optional, default is off).

- **Streamlit Cloud build too slow**:  
  Freeze versions in `requirements.txt`; avoid heavy extras unless needed.

---

## 13) Roadmap (next week+)

- **PINN / physics-informed surrogate**: small network to refine \(C_d\) using momentum balance priors; trained on synthetic + real Fluidyn data.  
- **Uncertainty**: quantile regressors or bootstrap ensembles to get prediction intervals.  
- **Active learning**: sample designs with high uncertainty → push to CFD queue → retrain.  
- **Explainability**: SHAP for global feature importance; per-prediction contributions.  
- **Case composer**: one-click export of 0/ fields + `fvSolution` hints from predicted BCs.

---

## 14) License & cost

- Code can be MIT/Apache-2.0 (your choice in the repo).  
- **Cost**: zero for local training/prediction. Streamlit Community Cloud is free for small apps. If you turn on commercial LLMs (OpenAI), that would incur cost — the app ships with **LLM OFF** by default.

---

## 15) FAQ

- **Do I need GPUs?** No. XGBoost/LightGBM CPU is fine for these sizes.  
- **How many samples?** Start with 800 for a quick POC; 3000 improves generalization.  
- **Can I swap models?** Yes, edit `train_surrogates.py` to use LightGBM or pure sklearn.  
- **How do I harden for production?** Pin package versions, add unit tests for data schema, and add model versioning + drift checks.

---

**Contact**: Ping with a training CSV to get a retraining command and demo plots for your next review.
