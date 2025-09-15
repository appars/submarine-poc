# Submarine Inlet BC & Drag Estimator (POC)

[**▶️ Live Demo on Streamlit**](https://submarine-poc-aw4sk3aafktkv9mmer4gkb.streamlit.app/)

Predict **inlet boundary conditions** (U_mag, TI, Lt → k, ω) and **drag coefficient (Cd)** from submarine design parameters using a **surrogate model** trained on **synthetic CFD-inspired data**. Includes explainability, a diagnostic “medical check report”, and a full metrics dashboard.

---

## ✨ Features

- **Inputs:** L, D, U, Re (derived), nose/tail shape factors, fin area ratio, AoA, depth  
- **Outputs:**  
  - Inlet BCs: **U_mag, TI, Lt** and derived **k, ω**  
  - **Cd** (surrogate)
- **Explainability:** “Explain my BCs” engineering rationale bullets  
- **Diagnostic health report:** “normal band vs actual” with ✅/⚠️/❌ + narrative (like a lab report)  
- **Metrics:** MAE, RMSE, MAPE (%), R², **IoU@5%** (tolerance overlap), latency & **efficiency** (R² per ms), learning curve, scatter  
- **Data:** Built-in synthetic data generator; CSV download & training from CSV  
- **Export:** One-click **inlet JSON** for OpenFOAM-style setups

---

## 🚀 Quick Start (Local)

> **Python:** 3.11 recommended (3.10–3.12 OK)

```bash
git clone https://github.com/appars/submarine-poc.git
cd submarine-poc

python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows powershell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Option A: let the app auto-train (defaults to 3000 samples)
export AUTO_TRAIN=1
export AUTO_TRAIN_SAMPLES=3000   # powershell: $env:AUTO_TRAIN_SAMPLES="3000"
streamlit run app.py
```

**Tip:** Commit a trained model (`models/surrogates.joblib`) to make the app open instantly anywhere (including Streamlit Cloud).

---

## 🧪 Train from a CSV and Save the Model

If you downloaded or generated a synthetic dataset (e.g., `data/samples_3000.csv`):

```bash
# still in the venv
python train_from_csv.py --csv data/samples_3000.csv
# -> writes models/surrogates.joblib
```

Commit it:
```bash
git add models/surrogates.joblib
git commit -m "Add trained surrogate (3k)"
git push
```

The app will now **load the model** and **skip training**.

---

## 🔧 Environment Variables

- `AUTO_TRAIN=1` — auto-train on first run if `models/surrogates.joblib` is missing  
- `AUTO_TRAIN_SAMPLES=3000` — number of synthetic rows used for training  
- (Optional) Metrics slider upper bound / default:
  - `METRICS_EVAL_MAX=3000`
  - `METRICS_EVAL_DEFAULT=3000`

---

## 📊 Model Metrics (what you’ll show)

- **Regression:** MAE, RMSE, MAPE (%), R²  
- **IoU@5%:** band-overlap metric treating ±5% tolerance as an interval around truth/prediction  
- **Latency:** ms/sample; **Efficiency:** R² / latency_ms  
- **Learning curve:** Cd MAE vs train size  
- **Scatter:** Cd truth vs prediction (+ CSV download)

You can bump the evaluation size up to **3000** in the UI (or via env vars above).

---

## 🩺 Diagnostic “Medical Check” Report

For each value it shows:

- **Normal band (≈)**: expected range computed from physics-inspired heuristics  
- **Actual**: the model’s current prediction  
- **Status**: ✅ OK, ⚠️ Slightly out, ❌ Out of range  
- A short **Impression** section that calls out outliers and suggests levers (AoA, tail taper, fins, meshing).

This gives non-ML colleagues an instant sanity check—like **BP 80/120 vs 85/125**.

---

## 📤 Inlet JSON Export

Click **Download inlet JSON** to get a minimal structure you can adapt in OpenFOAM:

```json
{
  "U_inlet":     {"type": "fixedValue", "value": "uniform (X 0 0)"},
  "k_inlet":     {"type": "fixedValue", "value": "uniform K"},
  "omega_inlet": {"type": "fixedValue", "value": "uniform W"}
}
```

Where `X`, `K`, `W` are filled with the predicted values.

---

## 🏗️ How it Works (POC)

- **Data:** Synthetic generator simulates plausible trends in turbulence and drag:
  - TI from nose/tail/AoA
  - Lt from L, tail taper, fins
  - k, ω from U_mag, TI, Lt (SST k–ω style formulas)
  - Cd ≈ skin-friction (Re) + form (shape, AoA, fins)
- **Models:** `HistGradientBoostingRegressor` pipelines (StandardScaler + HGB)  
- **Targets:** BC_U_mag, BC_TI, BC_Lt, and Cd  
- **Explainability:** narrative bullets + diagnostic bands
- **Swap-in real data:** Replace the CSV with Fluidyn/OpenFOAM samples → train and re-deploy

---

## 🗂️ Repo Layout

```
app.py                    # Streamlit app (auto-train, diagnostics, metrics)
requirements.txt
train_from_csv.py         # Train on provided CSV and save models/surrogates.joblib
models/
  └─ surrogates.joblib    # (optional) committed model for instant startup
data/
  └─ samples_*.csv        # (optional) your synthetic or real data
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (include `models/surrogates.joblib` for instant load).  
2. Go to **https://streamlit.io/cloud → New app**  
   - **Repo:** `appars/submarine-poc`  
   - **Branch:** `main`  
   - **Main file:** `app.py`  
3. **Advanced settings → Environment variables**  
   - `AUTO_TRAIN=1`  
   - `AUTO_TRAIN_SAMPLES=3000`  
4. **Deploy** → Share the URL (e.g., `https://submarine-poc-<you>.streamlit.app`)

> If you see a Python 3.13 environment warning, pin **3.11** by adding a `.python-version` file with `3.11.9`, commit, and redeploy.

---

## ⚠️ Notes & Limitations

- This is a **POC** trained on **synthetic** data; absolute values are illustrative.  
- Replace with **Fluidyn/OpenFOAM datasets** to calibrate and validate.  
- Streamlit Cloud is **CPU only** and **ephemeral**: if you rely on training at runtime, it retrains on cold starts unless you commit the model file.

---

## 🛠 Tech Stack

- Streamlit • scikit-learn (HistGradientBoosting) • pandas • numpy • joblib

---

## 📄 License

MIT (feel free to change for your org). Add a `LICENSE` file if needed.

---

## 🙋 Support

Open an issue or ping in Discussions if you want:
- a **one-page PDF** cheat sheet for the demo,
- a **batch trainer** for Fluidyn data,
- or a **Dockerfile** for on-prem runs.
