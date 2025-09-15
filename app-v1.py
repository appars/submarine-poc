import os, math, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="Submarine BC & Drag Estimator", layout="wide")
st.title("Submarine Inlet BC & Drag Estimator")
st.caption("Apparsamy Perumal")

MODELS_PATH = Path("models/surrogates.joblib")
AUTO_TRAIN = os.environ.get("AUTO_TRAIN", "1") == "1"  # cloud auto-train on first run

# ----------------- synthetic data generator -----------------
def generate_synth(n=300, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        L = float(rng.uniform(20.0, 80.0))
        D = float(rng.uniform(3.0, 10.0))
        U = float(rng.uniform(2.0, 15.0))
        rho = 1025.0
        mu  = 1.05e-3
        Re  = rho*U*L/mu
        nose = float(rng.uniform(0.2, 0.9))
        tail = float(rng.uniform(0.2, 0.95))
        fin = float(rng.uniform(0.00, 0.15))
        AoA = float(rng.uniform(-3.0, 3.0))
        depth = float(rng.uniform(10.0, 300.0))
        slenderness = L/D

        # BC heuristics (these act as "ground truth" for the POC)
        TI = 0.015 + 0.02*(1.0 - nose)*0.5 + 0.01*(1.0 - tail)*0.5 + 0.001*abs(AoA)
        TI = max(0.005, min(0.10, TI))
        Lt = 0.07 * L * (0.6 + 0.4*(1.0 - tail)) * (1.0 - 0.5*fin)
        Lt = max(0.01*D, min(0.3*L, Lt))
        U_mag = U * max(0.95, math.cos(math.radians(AoA)))
        k = 1.5*(U_mag*TI)**2
        Cmu = 0.09
        omega = (k**0.5) / ((Cmu**0.25)*Lt)

        # Cd surrogate
        if Re > 1e5:
            cf = 0.075 / ((math.log10(Re) - 2.0)**2)
        else:
            cf = 0.01
        cd_friction = cf * (2.0*slenderness)**-0.3
        cd_form = 0.002 + 0.004*(1.0 - tail) + 0.002*(1.0 - nose) + 0.02*(AoA/10.0)**2 + 0.01*fin
        Cd = max(0.0015, cd_friction + cd_form) * rng.normal(1.0, 0.02)

        rows.append(dict(
            L=L, D=D, speed_U=U, Re=Re, nose_shape_c=nose, tail_taper_c=tail,
            fin_area_ratio=fin, AoA_deg=AoA, depth_m=depth,
            BC_U_mag=U_mag, BC_TI=TI, BC_Lt=Lt, k=k, omega=omega, Cd=Cd
        ))
    return pd.DataFrame(rows)

# ----------------- trainer -----------------
def train_and_save(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    features = ["L","D","speed_U","Re","nose_shape_c","tail_taper_c","fin_area_ratio","AoA_deg","depth_m"]
    df["slenderness"] = df["L"]/df["D"]
    X = df[features + ["slenderness"]]
    y_bc = df[["BC_U_mag","BC_TI","BC_Lt"]]
    y_cd = df["Cd"]

    pre = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())], remainder="drop")
    params = dict(loss="squared_error", max_depth=6, learning_rate=0.06, max_iter=400, l2_regularization=0.0)

    bc_models = {}
    for tgt in y_bc.columns:
        pipe = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(**params))])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_bc[tgt], test_size=0.2, random_state=42)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        st.write(f"{tgt} MAE: {mean_absolute_error(y_te, pred):.5f}")
        bc_models[tgt] = pipe

    cd_model = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(**params))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_cd, test_size=0.2, random_state=42)
    cd_model.fit(X_tr, y_tr)
    st.write(f"Cd MAE: {mean_absolute_error(y_te, cd_model.predict(X_te)):.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bc_models": bc_models, "cd_model": cd_model, "feature_names": X.columns.tolist()}, out_path)

def ensure_model():
    if MODELS_PATH.exists():
        return
    if not AUTO_TRAIN:
        return
    with st.status("Training surrogate models online (first run only)...", state="running"):
        df = generate_synth(n=300)
        train_and_save(df, MODELS_PATH)
    st.success("Training complete. Model cached on server.")

# ----------------- load or auto-train -----------------
ensure_model()
if MODELS_PATH.exists():
    models = joblib.load(MODELS_PATH)
    bc_models = models["bc_models"]
    cd_model = models["cd_model"]
    feature_names = models["feature_names"]
    MODEL_SOURCE = "Model (gradient boosting)"
else:
    bc_models = cd_model = feature_names = None
    MODEL_SOURCE = "Heuristics (no trained model found)"

# ----------------- sidebar inputs -----------------
st.sidebar.header("Design Parameters")
L = st.sidebar.number_input("Length L (m)", 5.0, 120.0, 50.0, step=0.5)
D = st.sidebar.number_input("Diameter D (m)", 0.5, 15.0, 6.0, step=0.1)
U  = st.sidebar.number_input("Speed U (m/s)", 0.1, 25.0, 10.0, step=0.1)
rho = st.sidebar.number_input("Density ρ (kg/m³)", 900.0, 1100.0, 1025.0, step=1.0)
mu  = st.sidebar.number_input("Dyn. viscosity μ (Pa·s)", 8e-4, 2e-3, 1.05e-3, step=1e-5, format="%.6f")
Re  = rho * U * L / mu
st.sidebar.write(f"Re ≈ **{Re:,.2e}**")

nose_c = st.sidebar.slider("Nose shape (0 blunt → 1 sharp)", 0.0, 1.0, 0.6, 0.05)
tail_c = st.sidebar.slider("Tail taper (0 short → 1 long)", 0.0, 1.0, 0.7, 0.05)
fin_ar = st.sidebar.slider("Fin area ratio", 0.00, 0.20, 0.05, 0.01)
AoA    = st.sidebar.slider("Angle of attack (deg)", -5.0, 5.0, 0.0, 0.5)
depth  = st.sidebar.number_input("Depth (m)", 5.0, 500.0, 50.0, step=5.0)

st.markdown("Use **Predict**. If no model is trained, a heuristics fallback is used so you can still demo.")

# -------------- heuristic fallback --------------
def heuristics_predict(L, D, U, Re, nose_c, tail_c, fin_ar, AoA, depth):
    slenderness = L / D
    TI = 0.015 + 0.02*(1.0 - nose_c)*0.5 + 0.01*(1.0 - tail_c)*0.5 + 0.001*abs(AoA)
    TI = max(0.005, min(0.10, TI))
    Lt = 0.07 * L * (0.6 + 0.4*(1.0 - tail_c)) * (1.0 - 0.5*fin_ar)
    Lt = max(0.01*D, min(0.3*L, Lt))
    U_mag = U * max(0.95, math.cos(math.radians(AoA)))
    k = 1.5*(U_mag*TI)**2
    Cmu = 0.09
    omega = (k**0.5) / ((Cmu**0.25)*Lt)
    if Re > 1e5:
        cf = 0.075 / ((math.log10(Re) - 2.0)**2)
    else:
        cf = 0.01
    cd_friction = cf * (2.0*slenderness)**-0.3
    cd_form = 0.002 + 0.004*(1.0 - tail_c) + 0.002*(1.0 - nose_c) + 0.02*(AoA/10.0)**2 + 0.01*fin_ar
    Cd = max(0.0015, cd_friction + cd_form)
    return U_mag, TI, Lt, k, omega, Cd

# -------------- session_state --------------
for k,v in {"has_pred": False, "U_mag": None, "TI": None, "Lt": None, "k": None, "omega": None, "Cd": None, "source": None, "params": None}.items():
    st.session_state.setdefault(k, v)

# -------------- predict --------------
if st.button("Predict"):
    slenderness = L / D
    params = dict(L=L, D=D, U=U, Re=Re, nose_c=nose_c, tail_c=tail_c, fin_ar=fin_ar, AoA=AoA, depth=depth, slenderness=slenderness)

    if bc_models is not None and cd_model is not None:
        row = {
            "L": L, "D": D, "speed_U": U, "Re": Re,
            "nose_shape_c": nose_c, "tail_taper_c": tail_c,
            "fin_area_ratio": fin_ar, "AoA_deg": AoA, "depth_m": depth,
            "slenderness": slenderness,
        }
        X_df = pd.DataFrame([[row[c] for c in feature_names]], columns=feature_names)
        U_mag = float(bc_models["BC_U_mag"].predict(X_df))
        TI    = float(bc_models["BC_TI"].predict(X_df))
        Lt    = float(bc_models["BC_Lt"].predict(X_df))
        TI = max(0.001, min(0.15, TI))
        Lt = max(0.01*D, min(0.3*L, Lt))
        k = 1.5*(U_mag*TI)**2
        Cmu = 0.09
        omega = (k**0.5) / ((Cmu**0.25)*Lt)
        Cd = float(cd_model.predict(X_df))
        source = MODEL_SOURCE
    else:
        U_mag, TI, Lt, k, omega, Cd = heuristics_predict(**params)
        source = "Heuristics (no trained model found)"

    st.session_state.update(dict(
        has_pred=True, U_mag=U_mag, TI=TI, Lt=Lt, k=k, omega=omega, Cd=Cd,
        source=source, params=params
    ))

# -------------- show predictions --------------
if st.session_state.has_pred:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Inlet BCs (recommended)")
        st.metric("U_mag (m/s)", f"{st.session_state.U_mag:.3f}")
        st.metric("Turbulence Intensity TI", f"{st.session_state.TI:.4f}")
        st.metric("Turbulence Length Scale Lt (m)", f"{st.session_state.Lt:.3f}")
        st.code(f"k = {st.session_state.k:.6f}\nomega = {st.session_state.omega:.6f}")
    with c2:
        st.subheader("Drag Coefficient")
        st.metric("Cd (surrogate)", f"{st.session_state.Cd:.5f}")
        st.caption(f"Source: {st.session_state.source}")

    inlet = {
        "U_inlet": {"type": "fixedValue", "value": f"uniform ({st.session_state.U_mag:.6f} 0 0)"},
        "k_inlet": {"type": "fixedValue", "value": f"uniform {st.session_state.k:.6f}"},
        "omega_inlet": {"type": "fixedValue", "value": f"uniform {st.session_state.omega:.6f}"}
    }
    st.download_button("Download inlet JSON", json.dumps(inlet, indent=2), file_name="inlet_bc.json")
else:
    st.info("Click **Predict** to compute inlet BCs and Cd.")

# -------------- Explain my BCs (offline only on cloud) --------------
st.divider()
st.subheader("Explain my BCs")
if not st.session_state.get("has_pred", False):
    st.warning("Run **Predict** first, then generate an explanation.")
else:
    if st.checkbox("Generate an explanation for these predictions"):
        p = st.session_state.params
        bullets = [
            f"- Slenderness L/D={p['L']/p['D']:.2f} reduces form drag; sharper nose ({p['nose_c']:.2f}) and longer tail ({p['tail_c']:.2f}) keep TI and wake low.",
            f"- Re={p['Re']:.2e} ⇒ fully turbulent regime; TI={st.session_state.TI:.4f} and Lt={st.session_state.Lt:.3f} m suit SST k–ω at AoA={p['AoA']:.1f}°.",
            f"- U_mag={st.session_state.U_mag:.3f} m/s includes minor cosine loss from AoA; k={st.session_state.k:.5f}, ω={st.session_state.omega:.5f} stabilize inlet.",
            f"- Cd={st.session_state.Cd:.5f}: friction likely dominates; form drag climbs if tail tapers less, nose blunts, or AoA increases.",
            f"- Mesh: aim y+≈30–100 (wall functions); ≥10–15 inlet diameters; expansion <5%.",
            f"- Numerics: relaxed SIMPLE & bounded schemes; if residuals stall, lower under-relaxation or refine suspected separation zones.",
            f"- Watch-outs: ↑AoA / ↑fin area → higher TI & secondary flows; refine around appendages/junctions.",
        ]
        st.markdown("\n".join(bullets))

# ======================= METRICS PANEL =======================
st.divider()
st.subheader("Model Metrics (Eval on synthetic holdout)")

def mean_iou_tolerance(y_true, y_pred, rel_band=0.05, abs_floor=1e-9):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    band_t = np.maximum(rel_band*np.abs(y_true), abs_floor)
    band_p = np.maximum(rel_band*np.abs(y_pred), abs_floor)
    d = np.abs(y_true - y_pred)
    inter = np.maximum(0.0, band_t + band_p - d)
    union = band_t + band_p + d
    return float(np.mean(inter / union))

n_eval = st.slider("Evaluation set size", 100, 800, 300, 50)
if st.button("Run metrics"):
    with st.status("Generating evaluation set & computing metrics…", state="running"):
        df_eval = generate_synth(n=n_eval, seed=123)

        # Build X and y
        features = ["L","D","speed_U","Re","nose_shape_c","tail_taper_c","fin_area_ratio","AoA_deg","depth_m"]
        df_eval["slenderness"] = df_eval["L"]/df_eval["D"]
        X_eval = df_eval[features + ["slenderness"]]

        y_true = {
            "BC_U_mag": df_eval["BC_U_mag"].values,
            "BC_TI": df_eval["BC_TI"].values,
            "BC_Lt": df_eval["BC_Lt"].values,
            "Cd": df_eval["Cd"].values,
        }

        # Predict and time latency
        t0 = time.perf_counter()
        if bc_models is not None and cd_model is not None:
            U_hat = bc_models["BC_U_mag"].predict(X_eval)
            TI_hat = bc_models["BC_TI"].predict(X_eval)
            Lt_hat = bc_models["BC_Lt"].predict(X_eval)
            Cd_hat = cd_model.predict(X_eval)
            src = "Model"
        else:
            # Fallback: compute with heuristics function (latency timing still measured)
            preds = [heuristics_predict(df_eval["L"][i], df_eval["D"][i], df_eval["speed_U"][i],
                                        df_eval["Re"][i], df_eval["nose_shape_c"][i], df_eval["tail_taper_c"][i],
                                        df_eval["fin_area_ratio"][i], df_eval["AoA_deg"][i], df_eval["depth_m"][i])
                     for i in range(len(df_eval))]
            U_hat, TI_hat, Lt_hat, _, _, Cd_hat = map(np.array, zip(*preds))
            src = "Heuristics"
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0 / len(df_eval)

        y_pred = {"BC_U_mag": U_hat, "BC_TI": TI_hat, "BC_Lt": Lt_hat, "Cd": Cd_hat}

        # Metrics function
        def reg_metrics(y, yhat):
            mae = mean_absolute_error(y, yhat)
            rmse = math.sqrt(mean_squared_error(y, yhat))
            mape = float(np.mean(np.abs((np.array(y) - np.array(yhat)) / (np.array(y) + 1e-12)))) * 100.0
            r2 = r2_score(y, yhat)
            iou = mean_iou_tolerance(y, yhat, rel_band=0.05)
            return dict(MAE=mae, RMSE=rmse, MAPE_pct=mape, R2=r2, IoU_5pct=iou)

        # Build metrics tables
        bc_summary = {}
        for k in ["BC_U_mag", "BC_TI", "BC_Lt"]:
            bc_summary[k] = reg_metrics(y_true[k], y_pred[k])
        cd_summary = reg_metrics(y_true["Cd"], y_pred["Cd"])

        # Cost efficiency (higher is better): R2 per millisecond
        eff_cd = cd_summary["R2"] / max(latency_ms, 1e-6)
        eff_bc = np.mean([bc_summary[k]["R2"] for k in bc_summary]) / max(latency_ms, 1e-6)

        # Learning curve / "loss log" for Cd: train on increasing sizes
        sizes = np.linspace(0.2, 1.0, 6)  # 20%..100% of 300
        lc_sizes, lc_mae = [], []
        df_lc = generate_synth(n=300, seed=777)
        df_lc["slenderness"] = df_lc["L"]/df_lc["D"]
        X_lc = df_lc[features + ["slenderness"]]
        y_lc = df_lc["Cd"]
        for frac in sizes:
            n_train = max(40, int(len(df_lc)*frac))
            X_tr, X_te, y_tr, y_te = train_test_split(X_lc, y_lc, test_size=0.2, random_state=int(1000*frac))
            # fit on n_train from X_tr
            X_tr_small = X_tr.iloc[:n_train].copy()
            y_tr_small = y_tr.iloc[:n_train].copy()
            pre = ColumnTransformer([("num", StandardScaler(), X_tr_small.columns.tolist())], remainder="drop")
            model = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=400))])
            model.fit(X_tr_small, y_tr_small)
            mae = mean_absolute_error(y_te, model.predict(X_te))
            lc_sizes.append(n_train); lc_mae.append(mae)

        # Cache to session for display
        st.session_state.metrics = dict(
            src=src, n_eval=n_eval, latency_ms=latency_ms,
            bc_summary=bc_summary, cd_summary=cd_summary,
            eff_cd=eff_cd, eff_bc=eff_bc,
            lc_sizes=lc_sizes, lc_mae=lc_mae,
            eval_table=pd.DataFrame({
                "BC_U_mag_true": y_true["BC_U_mag"], "BC_U_mag_pred": y_pred["BC_U_mag"],
                "BC_TI_true": y_true["BC_TI"], "BC_TI_pred": y_pred["BC_TI"],
                "BC_Lt_true": y_true["BC_Lt"], "BC_Lt_pred": y_pred["BC_Lt"],
                "Cd_true": y_true["Cd"], "Cd_pred": y_pred["Cd"],
            })
        )

    st.success("Metrics computed.")

# ---- render metrics if present ----
m = st.session_state.get("metrics")
if m:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Eval samples", m["n_eval"])
        st.metric("Avg latency (ms/sample)", f"{m['latency_ms']:.3f}")
    with c2:
        st.metric("Cd R²", f"{m['cd_summary']['R2']:.4f}")
        st.metric("Cd IoU@5%", f"{m['cd_summary']['IoU_5pct']:.4f}")
    with c3:
        st.metric("Efficiency (Cd)", f"{m['eff_cd']:.3f}")
        st.caption("Efficiency = R² / latency_ms")

    st.markdown("### Regression Metrics — BCs")
    st.dataframe(pd.DataFrame(m["bc_summary"]).T.style.format({
        "MAE":"{:.5f}", "RMSE":"{:.5f}", "MAPE_pct":"{:.3f}", "R2":"{:.4f}", "IoU_5pct":"{:.4f}"
    }))

    st.markdown("### Regression Metrics — Cd")
    st.dataframe(pd.DataFrame([m["cd_summary"]], index=["Cd"]).style.format({
        "MAE":"{:.5f}", "RMSE":"{:.5f}", "MAPE_pct":"{:.3f}", "R2":"{:.4f}", "IoU_5pct":"{:.4f}"
    }))

    st.markdown("### Loss log / Learning curve (Cd MAE vs. train size)")
    lc_df = pd.DataFrame({"Train size": m["lc_sizes"], "MAE": m["lc_mae"]})
    st.line_chart(lc_df, x="Train size", y="MAE", height=240)

    st.markdown("### Predictions vs. Truth (Cd)")
    scatter_df = pd.DataFrame({"Truth": m["eval_table"]["Cd_true"], "Pred": m["eval_table"]["Cd_pred"]})
    st.scatter_chart(scatter_df, x="Truth", y="Pred", height=300)
    st.caption("Ideal = diagonal line; tighter clustering ⇒ better surrogate.")

    # Download eval CSV
    csv = m["eval_table"].to_csv(index=False).encode("utf-8")
    st.download_button("Download eval predictions (CSV)", csv, file_name="eval_predictions.csv", mime="text/csv")

st.divider()
st.caption("Tip: commit models/surrogates.joblib to skip first-run training. AUTO_TRAIN=1 retrains in-cloud if model is missing.")

