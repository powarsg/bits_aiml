#!/usr/bin/env python3
"""
bike_compare_and_train.py

- Uses KFold (shuffle=True, random_state=42)
- Compares CatBoost trained on raw target vs log1p(target) using CV RMSLE
- Trains final model on full data with winning approach
- Produces submission.csv with original test datetime strings and column 'count_predicted'
- Saves residual and feature importance plots
"""

import os
os.environ["CATBOOST_VERBOSE_LOGGING"] = "False"

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("bike_pipeline")

# -----------------------
# Utility: RMSLE
# -----------------------
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# -----------------------
# Feature engineering (same as your pipeline)
# -----------------------
def preprocess(df, is_train=True):
    df = df.copy()
    log.info(f"Preprocessing started: is_train={is_train}, shape={df.shape}")

    # Parse datetime differently for train vs test
    if is_train:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")

    # Basic date features
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Interaction features (guarded)
    if "temp" in df.columns and "atemp" in df.columns:
        df["temp_atemp_diff"] = df["atemp"] - df["temp"]
    if "humidity" in df.columns and "windspeed" in df.columns:
        df["humid_wind"] = df["humidity"] * df["windspeed"]
    if "windspeed" in df.columns:
        df["wind_is_zero"] = (df["windspeed"] == 0).astype(int)

    # Drop original datetime
    df = df.drop(columns=["datetime"])

    # Drop leak columns in train
    if is_train:
        df = df.drop(columns=["casual", "registered"], errors="ignore")

    log.info(f"Preprocessing completed: new shape={df.shape}")
    return df

# -----------------------
# Load data
# -----------------------
log.info("Loading datasets...")
train = pd.read_csv("bike_train.csv")
test = pd.read_csv("bike_test.csv")

log.info(f"Raw train shape: {train.shape}")
log.info(f"Raw test shape: {test.shape}")
log.info(f"Train columns: {list(train.columns)}")
log.info(f"Test columns: {list(test.columns)}")

original_test_datetime = test["datetime"].astype(str).copy()

# -----------------------
# Preprocess
# -----------------------
train_proc = preprocess(train, is_train=True)
test_proc = preprocess(test, is_train=False)

# split features/target
if "count" not in train_proc.columns:
    raise ValueError("Training data must include 'count' column.")

X = train_proc.drop(columns=["count"])
y = train_proc["count"]

# ensure test has same columns (if any minor mismatch, align and fill zeros)
missing_cols = [c for c in X.columns if c not in test_proc.columns]
if missing_cols:
    log.warning("Test missing columns found; adding them with zeros: %s", missing_cols)
    for c in missing_cols:
        test_proc[c] = 0
extra_test_cols = [c for c in test_proc.columns if c not in X.columns]
if extra_test_cols:
    log.warning("Dropping extra test columns not in train: %s", extra_test_cols)
    test_proc = test_proc.drop(columns=extra_test_cols)

test_proc = test_proc[X.columns]  # align column order

log.info(f"Final feature count: {X.shape[1]}  ;  number of training rows: {X.shape[0]}")

# -----------------------
# Model params (recommended tuned baseline)
# -----------------------
base_params = {
    "depth": 8,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5,
    "bagging_temperature": 0.5,
    "random_strength": 1.0,
    "n_estimators": 3000,
    "loss_function": "RMSE",
    "random_seed": 42,
    "od_type": "Iter",
    "od_wait": 60,
    "verbose": False
}

# -----------------------
# CV compare: raw vs log1p targets
# -----------------------
log.info("Starting CV comparison: raw target vs log1p(target)")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores_raw = []
scores_log = []

# store OOF preds for diagnostic plots
oof_raw = np.zeros(len(X))
oof_log = np.zeros(len(X))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
    log.info("Fold %d: train rows=%d, val rows=%d", fold, len(tr_idx), len(val_idx))

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # --- raw model ---
    mraw = CatBoostRegressor(**base_params)
    mraw.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    pred_raw = mraw.predict(X_val)
    pred_raw = np.maximum(0, pred_raw)
    score_raw = rmsle(y_val.values, pred_raw)
    scores_raw.append(score_raw)
    oof_raw[val_idx] = pred_raw

    # --- log model ---
    mlog = CatBoostRegressor(**base_params)
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)
    mlog.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log), use_best_model=True)
    pred_log = mlog.predict(X_val)            # in log1p space
    pred_log_inv = np.expm1(pred_log)
    pred_log_inv = np.maximum(0, pred_log_inv)
    score_log = rmsle(y_val.values, pred_log_inv)
    scores_log.append(score_log)
    oof_log[val_idx] = pred_log_inv

    log.info("Fold %d results -> raw RMSLE: %.6f | log1p RMSLE: %.6f", fold, score_raw, score_log)

mean_raw = float(np.mean(scores_raw))
mean_log = float(np.mean(scores_log))

log.info("CV mean RMSLE -> raw: %.6f | log1p: %.6f", mean_raw, mean_log)

# Decide winner
if mean_log < mean_raw:
    winner = "log1p"
    log.info("Winner: train on log1p(target). Will train final model on log1p and invert predictions.")
else:
    winner = "raw"
    log.info("Winner: train on raw target. Will train final model on raw counts.")

# -----------------------
# Diagnostic residual plots for both approaches (combined OOF)
# -----------------------
plt.figure(figsize=(8,5))
plt.scatter(y, oof_raw - y, alpha=0.3, s=10)
plt.axhline(0, color='red', linewidth=1)
plt.xlabel("True Count")
plt.ylabel("Residual (raw_pred - true)")
plt.title("OOF Residuals (raw target)")
plt.tight_layout()
plt.savefig("oof_residuals_raw.png")
plt.close()
log.info("Saved oof_residuals_raw.png")

plt.figure(figsize=(8,5))
plt.scatter(y, oof_log - y, alpha=0.3, s=10)
plt.axhline(0, color='red', linewidth=1)
plt.xlabel("True Count")
plt.ylabel("Residual (log1p_inv_pred - true)")
plt.title("OOF Residuals (log1p target)")
plt.tight_layout()
plt.savefig("oof_residuals_log1p.png")
plt.close()
log.info("Saved oof_residuals_log1p.png")

# -----------------------
# Train final model on full dataset using winner
# -----------------------
log.info("Training final model on FULL dataset using: %s", winner)

final_model = CatBoostRegressor(**base_params)

if winner == "raw":
    final_model.fit(X, y, verbose=False)
    test_pred = final_model.predict(test_proc)
    test_pred = np.maximum(0, test_pred)
else:
    final_model.fit(X, np.log1p(y), verbose=False)
    test_pred_log = final_model.predict(test_proc)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.maximum(0, test_pred)

# -----------------------
# Feature importance plot (final model)
# -----------------------
try:
    importances = final_model.get_feature_importance()
    names = X.columns
    idx = np.argsort(importances)[::-1]
    top_n = min(len(names), 30)
    plt.figure(figsize=(8, min(0.3*top_n+2, 12)))
    plt.barh(np.array(names)[idx][:top_n][::-1], importances[idx][:top_n][::-1])
    plt.title("Feature importances (final model)")
    plt.tight_layout()
    plt.savefig("feature_importance_final.png")
    plt.close()
    log.info("Saved feature_importance_final.png")
except Exception as e:
    log.warning("Could not create feature importance plot: %s", str(e))

# -----------------------
# Final residual plot using OOF predictions of the chosen approach
# -----------------------
if winner == "raw":
    final_oof = oof_raw
else:
    final_oof = oof_log

plt.figure(figsize=(8,5))
plt.scatter(y, final_oof - y, alpha=0.3, s=10)
plt.axhline(0, color='red', linewidth=1)
plt.xlabel("True Count")
plt.ylabel("Residual (pred - true)")
plt.title(f"OOF Residuals (chosen: {winner})")
plt.tight_layout()
plt.savefig("oof_residuals_chosen.png")
plt.close()
log.info("Saved oof_residuals_chosen.png")

# -----------------------
# Submission
# -----------------------
submission = pd.DataFrame({
    "datetime": original_test_datetime,
    "count_predicted": np.round(test_pred).astype(int)
})

submission.to_csv("submission.csv", index=False)
log.info("Saved submission.csv (original test datetime strings preserved)")

# -----------------------
# Final summary log
# -----------------------
log.info("===== SUMMARY =====")
log.info("CV mean RMSLE (raw)  : %.6f", mean_raw)
log.info("CV mean RMSLE (log1p): %.6f", mean_log)
log.info("Selected method       : %s", winner)
log.info("Submission saved to   : submission.csv")
log.info("Plots saved: oof_residuals_raw.png, oof_residuals_log1p.png, oof_residuals_chosen.png, feature_importance_final.png (if available)")
log.info("Done.")
