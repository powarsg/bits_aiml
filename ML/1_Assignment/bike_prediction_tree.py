import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# ---------------------------------------------------
# Logging setup
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger()

# silence CatBoost internal logs fully
import os
os.environ["CATBOOST_VERBOSE_LOGGING"] = "False"


# ---------------------------------------------------
# RMSLE function
# ---------------------------------------------------
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
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

    # Interaction features
    if "temp" in df.columns and "atemp" in df.columns:
        df["temp_atemp_diff"] = df["atemp"] - df["temp"]

    if "humidity" in df.columns and "windspeed" in df.columns:
        df["humid_wind"] = df["humidity"] * df["windspeed"]

    if "windspeed" in df.columns:
        df["wind_is_zero"] = (df["windspeed"] == 0).astype(int)

    # Remove original datetime column
    df = df.drop(columns=["datetime"])

    # Drop columns not in test
    if is_train:
        df = df.drop(columns=["casual", "registered"], errors="ignore")

    log.info(f"Preprocessing completed: new shape={df.shape}")
    log.info(f"Columns after preprocessing: {list(df.columns)}")

    return df


# ---------------------------------------------------
# Load data
# ---------------------------------------------------
log.info("Loading datasets...")
train = pd.read_csv("bike_train.csv")
test = pd.read_csv("bike_test.csv")

log.info(f"Train shape before preprocessing: {train.shape}")
log.info(f"Test shape before preprocessing: {test.shape}")
log.info(f"Train columns: {list(train.columns)}")
log.info(f"Test columns: {list(test.columns)}")

original_test_datetime = test["datetime"].copy()

# ---------------------------------------------------
# Preprocess
# ---------------------------------------------------
log.info("Preprocessing training dataset...")
train_proc = preprocess(train, is_train=True)

log.info("Preprocessing test dataset...")
test_proc = preprocess(test, is_train=False)

# ---------------------------------------------------
# Split features + target
# ---------------------------------------------------
X = train_proc.drop(columns=["count"])
y = train_proc["count"]

# ensure same column order
test_proc = test_proc[X.columns]

log.info(f"Final Train X shape: {X.shape}")
log.info(f"Final Train y shape: {y.shape}")
log.info(f"Final Test shape: {test_proc.shape}")


# ---------------------------------------------------
# Train CatBoost (no logs)
# ---------------------------------------------------
log.info("Training CatBoost model...")

model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    n_estimators=1500,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_preds = []
scores = []
all_val_true = []
all_val_pred = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    log.info(f"--------------- Fold {fold+1} started ---------------")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    log.info(f"Train fold shape: {X_tr.shape}, Validation fold shape: {X_val.shape}")

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    val_pred = model.predict(X_val)
    val_pred = np.maximum(0, val_pred)

    fold_rmsle = rmsle(y_val, val_pred)
    scores.append(fold_rmsle)

    log.info(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

    all_val_true.append(y_val)
    all_val_pred.append(val_pred)

    # Residual plot for this fold
    plt.figure(figsize=(7, 4))
    plt.scatter(y_val, y_val - val_pred, alpha=0.4)
    plt.axhline(0, color='red')
    plt.xlabel("Actual Count")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - Fold {fold+1}")
    #plt.tight_layout()
    #plt.savefig(f"residuals_fold_{fold+1}.png")
    #plt.close()

log.info(f"AVG RMSLE: {np.mean(scores):.5f}")


# ---------------------------------------------------
# Global residual plot
# ---------------------------------------------------
all_true = np.concatenate(all_val_true)
all_pred = np.concatenate(all_val_pred)

plt.figure(figsize=(7, 4))
plt.scatter(all_true, all_true - all_pred, alpha=0.3)
plt.axhline(0, color='red')
plt.xlabel("Actual Count")
plt.ylabel("Residuals")
plt.title("Residual Plot (All Folds Combined)")
plt.tight_layout()
plt.savefig("residuals_all_folds.png")
plt.close()


# ---------------------------------------------------
# Feature Importance Plot
# ---------------------------------------------------
log.info("Plotting feature importances...")

importances = model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 12))
indices = np.argsort(importances)
plt.barh(range(len(importances)), importances[indices])
plt.yticks(range(len(importances)), feature_names[indices])
plt.xlabel("Importance Score")
plt.title("CatBoost Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance_catboost.png")
plt.close()

# =====================================================
# Train Final Model on FULL data
# =====================================================
log.info("Training CatBoost on FULL dataset...")
model.fit(X, y, verbose=False)

# ---------------------------------------------------
# Predict test data
# ---------------------------------------------------
log.info("Predicting test data...")
test_pred = model.predict(test_proc)
test_pred = np.maximum(0, test_pred)


# ---------------------------------------------------
# Prepare submission
# ---------------------------------------------------
submission = pd.DataFrame({
    "datetime": original_test_datetime,
    "count_predicted": test_pred.round().astype(int)
})

submission.to_csv("submission_cat.csv", index=False)
log.info("Submission file created successfully: submission_cat.csv")
log.info("Residual plots and feature importance charts saved.")
