# train_from_csv.py
import argparse, math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["L","D","speed_U","Re","nose_shape_c","tail_taper_c","fin_area_ratio","AoA_deg","depth_m"]
TARGETS_BC = ["BC_U_mag","BC_TI","BC_Lt"]
TARGET_CD = "Cd"

def train_and_save(df: pd.DataFrame, out_path: Path, random_state=42):
    # Construct features (with slenderness to match app.py expectations)
    df = df.copy()
    df["slenderness"] = df["L"] / df["D"]
    X = df[FEATURES + ["slenderness"]]

    # Split targets
    y_bc = df[TARGETS_BC]
    y_cd = df[TARGET_CD]

    # Same preprocessing/model class as the app
    pre = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())], remainder="drop")
    hgb_params = dict(loss="squared_error", max_depth=6, learning_rate=0.06, max_iter=400, l2_regularization=0.0)

    # Train BC models
    bc_models = {}
    print("\n--- Training inlet BC regressors ---")
    for tgt in TARGETS_BC:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_bc[tgt], test_size=0.2, random_state=random_state)
        pipe = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(**hgb_params))])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        rmse = mean_squared_error(y_te, pred, squared=False)
        r2 = r2_score(y_te, pred)
        print(f"{tgt}: MAE={mae:.5f}  RMSE={rmse:.5f}  R2={r2:.4f}")
        bc_models[tgt] = pipe

    # Train Cd model
    print("\n--- Training Cd regressor ---")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_cd, test_size=0.2, random_state=random_state)
    cd_model = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(**hgb_params))])
    cd_model.fit(X_tr, y_tr)
    pred_cd = cd_model.predict(X_te)
    mae = mean_absolute_error(y_te, pred_cd)
    rmse = mean_squared_error(y_te, pred_cd, squared=False)
    r2 = r2_score(y_te, pred_cd)
    print(f"Cd  : MAE={mae:.5f}  RMSE={rmse:.5f}  R2={r2:.4f}")

    # Save exactly in the format the app expects
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"bc_models": bc_models, "cd_model": cd_model, "feature_names": X.columns.tolist()}
    joblib.dump(payload, out_path)
    print(f"\nSaved model bundle -> {out_path.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to synthetic CSV (e.g., data/samples_3000.csv)")
    ap.add_argument("--out", default="models/surrogates.joblib", help="Where to save the trained bundle")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # quick validation that necessary columns exist
    needed = set(FEATURES + TARGETS_BC + [TARGET_CD])
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")
    train_and_save(df, Path(args.out))

