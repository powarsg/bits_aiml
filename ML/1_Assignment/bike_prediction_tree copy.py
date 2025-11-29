import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# =====================================================
# Logging Configuration
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger()

# Silence CatBoost internal logs
os.environ["CATBOOST_VERBOSE_LOGGING"] = "False"


# =====================================================
# RMSLE
# =====================================================
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))


# =====================================================
# Preprocessing + Feature Engineering
# =====================================================
def preprocess(df, is_train=True):
    df = df.copy()

    # Parse datetime based on dataset type
    if is_train:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")

    # Date-based features
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Interaction features
    df["temp_atemp_diff"] = df["atemp"] - df["temp"]
    df["humid_wind"] = df["humidity"] * df["windspeed"]
    df["wind_is_zero"] = (df["windspeed"] == 0).astype(int)

    # Drop columns not used
    df = df.drop(columns=["datetime"])

    # Remove train-only columns
    if is_train:
        df = df.drop(columns=["casual", "registered"], errors="ignore")

    return df


# =====================================================
# Load Data
# =====================================================
log.info("Loading datasets...")
train = pd.read_csv("bike_train.csv")
test = pd.read_csv("bike_test.csv")

log.info(f"Training shape: {train.shape}, Test shape: {test.shape}")
log.info(f"Train columns: {list(train.columns)}")
log.info(f"Test columns: {list(test.columns)}")

original_test_datetime = test["datetime"]


# =====================================================
# Preprocess
# =====================================================
log.info("Preprocessing training data...")
train_proc = preprocess(train, is_train=True)

log.info("Preprocessing test data...")
test_proc = preprocess(test, is_train=False)

log.info(f"Processed train shape: {train_proc.shape}")
log.info(f"Processed test shape: {test_proc.shape}")


# =====================================================
# Train/Target Split
# =====================================================
X = train_proc.drop(columns=["count"])
y = train_proc["count"]

# Ensure test has same columns
test_proc = test_proc[X.columns]

log.info(f"Final feature count: {len(X.columns)}")


# =====================================================
# Model Configuration (CatBoost)
# =====================================================
model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    n_estimators=1500,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)


# =====================================================
# 5-Fold Cross Validation
# =====================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

oof_preds = np.zeros(len(X))

log.info("Starting 5-Fold CV...")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    log.info(f"========= Fold {fold+1} =========")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    log.info(f"Train fold shape: {X_tr.shape}, Validation fold shape: {X_val.shape}")

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    fold_rmsle = rmsle(y_val, val_pred)
    scores.append(fold_rmsle)

    log.info(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

log.info(f"===== Mean CV RMSLE: {np.mean(scores):.5f} =====")


# =====================================================
# Residual Plot
# =====================================================
plt.figure(figsize=(8, 5))
plt.scatter(y, oof_preds - y, alpha=0.4)
plt.axhline(0, color="red")
plt.title("Residuals vs True Count")
plt.xlabel("True Count")
plt.ylabel("Residual (Prediction - True)")
plt.savefig("residuals_plot.png")
log.info("Residual plot saved: residuals_plot.png")


# =====================================================
# Train Final Model on FULL data
# =====================================================
log.info("Training CatBoost on FULL dataset...")
model.fit(X, y, verbose=False)


# =====================================================
# Predict Test
# =====================================================
log.info("Predicting test data...")
test_pred = model.predict(test_proc)
test_pred = np.maximum(0, test_pred)

# =====================================================
# Feature Importance Plot
# =====================================================
importances = model.get_feature_importance()
sorted_idx = np.argsort(importances)[::-1]
top_idx = sorted_idx[:20]  # plot top 20

plt.figure(figsize=(8, 10))
plt.barh(np.array(X.columns)[top_idx], importances[top_idx])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (CatBoost)")
plt.savefig("feature_importance.png")
log.info("Feature importance plot saved: feature_importance.png")


# =====================================================
# Save Submission
# =====================================================
submission = pd.DataFrame({
    "datetime": original_test_datetime,
    "count_predicted": test_pred
})

submission.to_csv("submission_catboost.csv", index=False)
log.info("Submission file saved as: submission_catboost.csv")
