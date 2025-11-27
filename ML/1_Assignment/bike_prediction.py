# kaggle_final_pipeline.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Config / feature lists
# ----------------------------
TRAIN_FILE = "bike_train.csv"
TEST_FILE = "bike_test.csv"

# feature groups (as you've been using)
numeric_poly = ["temp", "atemp"]   # will get PolynomialFeatures(degree=2)
numeric_other = [
    "humidity", "windspeed",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "temp_humidity", "feels_like_diff"
]
categorical = [
    "season", "weather", "holiday", "workingday",
    "year", "is_weekend", "is_rush_hour"
]

FEATURES = numeric_poly + numeric_other + categorical

# ----------------------------
# Robust datetime parser (handles both formats)
# ----------------------------
def parse_datetime_series(s):
    # try train format, then test format, then let pandas infer
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M"):
        try:
            parsed = pd.to_datetime(s.astype(str), format=fmt, errors="coerce")
            if parsed.notna().sum() > 0:
                # if at least some parsed, return parsed (remaining NaT handle later)
                return parsed
        except Exception:
            pass
    # fallback: auto-parse (slower)
    return pd.to_datetime(s, errors="coerce")

# ----------------------------
# Feature engineering (used for train and test)
# ----------------------------
def add_datetime_features(df):
    df = df.copy()
    df["datetime"] = parse_datetime_series(df["datetime"])
    if df["datetime"].isna().any():
        # If any unparsable rows remain, raise to catch problem early
        nbad = df["datetime"].isna().sum()
        raise ValueError(f"{nbad} datetime rows could not be parsed. Check formats.")
    # basic parts
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    # cyclic encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # binary features
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # interactions
    # ensure numeric columns exist before computing
    df["temp_humidity"] = df["temp"] * df["humidity"]
    df["feels_like_diff"] = df["atemp"] - df["temp"]

    return df

# ----------------------------
# RMSLE helper
# ----------------------------
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# ----------------------------
# Load train, engineer features
# ----------------------------
train = pd.read_csv(TRAIN_FILE)
train = add_datetime_features(train)

# Clip category ranges to safe bounds (Kaggle dataset known ranges)
train["season"] = train["season"].clip(1, 4).astype(int)
train["weather"] = train["weather"].clip(1, 4).astype(int)
train["holiday"] = train["holiday"].astype(int)
train["workingday"] = train["workingday"].astype(int)

# keep only needed columns for model
X_all = train[FEATURES].copy()
y_all = train["count"].copy()

# ----------------------------
# Time-order split (80/20) for validation
# ----------------------------
train_sorted = train.sort_values("datetime").reset_index(drop=True)
split_index = int(len(train_sorted) * 0.8)

X_train = train_sorted.loc[: split_index - 1, FEATURES].copy()
y_train = train_sorted.loc[: split_index - 1, "count"].copy()

X_valid = train_sorted.loc[split_index:, FEATURES].copy()
y_valid = train_sorted.loc[split_index:, "count"].copy()

print(f"Train rows: {len(X_train)}, Valid rows: {len(X_valid)}")

# ----------------------------
# Preprocessing: polynomial on temp/atemp, scale others, OHE for categoricals
# Note: we will fit this only on training data and reuse for validation/test
# ----------------------------
preprocess = ColumnTransformer(transformers=[
    ("poly", PolynomialFeatures(degree=2, include_bias=False), numeric_poly),
    ("num", StandardScaler(), numeric_other),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
], remainder="drop")

# Fit preprocess on X_train to learn scalers & categories
preprocess.fit(X_train)

# Fill any missing numeric values in train/valid using training medians (stable)
train_median = X_train[numeric_poly + numeric_other].median()
X_train[numeric_poly + numeric_other] = X_train[numeric_poly + numeric_other].fillna(train_median)
X_valid[numeric_poly + numeric_other] = X_valid[numeric_poly + numeric_other].fillna(train_median)

# ----------------------------
# Models to train & evaluate (we will create pipelines)
# ----------------------------
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=20000),
    "RandomForest": RandomForestRegressor(
        n_estimators=800, max_depth=18, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
}

results = {}

# Train & evaluate on time-split validation
for name, base_model in models.items():
    pipe = Pipeline([
        ("pre", preprocess),
        ("model", base_model)
    ])
    pipe.fit(X_train, y_train)

    # validation predictions
    yv = pipe.predict(X_valid)
    score_rmsle = rmsle(y_valid, yv)
    score_r2 = r2_score(y_valid, yv)
    results[name] = {"RMSLE_valid": score_rmsle, "R2_valid": score_r2}
    #print(f"{name:12s}  RMSLE_valid={score_rmsle:.5f}  R2_valid={score_r2:.5f}")

print("\nValidation results:")
print(pd.DataFrame(results).T)

# ----------------------------
# Final: retrain on FULL training data, then predict test
# ----------------------------
# Prepare final preprocess fit on full X_all (fit again to include all categories / scales)
# But safer approach: fit preprocess on X_all to include all categories
preprocess_full = ColumnTransformer(transformers=[
    ("poly", PolynomialFeatures(degree=2, include_bias=False), numeric_poly),
    ("num", StandardScaler(), numeric_other),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
], remainder="drop")
preprocess_full.fit(X_all.fillna(train_median))

# Ensure we prepare test dataframe with same engineered columns
test = pd.read_csv(TEST_FILE)
datetime_backup = test["datetime"].astype(str)  # preserve original string formatting for submission
test = add_datetime_features(test)

# Clip categories and fill missing numerics
test["season"] = test["season"].clip(1, 4).astype(int)
test["weather"] = test["weather"].clip(1, 4).astype(int)
test["holiday"] = test["holiday"].astype(int)
test["workingday"] = test["workingday"].astype(int)

# fill numeric missing with train medians
test[numeric_poly + numeric_other] = test[numeric_poly + numeric_other].fillna(train_median)

X_test_final = test[FEATURES].copy()

# produce submissions per model (retrain each on full data)
for name, base_model in models.items():
    pipe_full = Pipeline([
        ("pre", preprocess_full),
        ("model", base_model)
    ])
    # retrain on entire training set (X_all)
    pipe_full.fit(X_all.fillna(train_median), y_all)

    preds = pipe_full.predict(X_test_final)
    preds = np.maximum(0, preds)       # no negatives
    preds_rounded = np.round(preds).astype(int)

    sub_df = pd.DataFrame({
        "datetime": datetime_backup,
        "count_predicted": preds_rounded
    })

    out_filename = f"submission_{name}.csv"
    sub_df.to_csv(out_filename, index=False)
    print(f"Wrote {out_filename}")

print("\nAll done. Check the CSVs and upload the one you prefer to Kaggle.")




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(name, numeric_poly, numeric_other, categorical, X_train, X_test, y_train):
    # 1. Choose a model
    #model_name = "Random Forest"   # or "XGBoost"
    model_name = name
    #model = models_all[model_name] # get the model object
    
    # 2. Fit the model on preprocessed data
    # Transform features only, to get feature matrix
    X_train_transformed = preprocess.fit_transform(X_train)
    
    # Fit the model on transformed data
    model_clone = model
    model_clone.fit(X_train_transformed, y_train)
    
    # 3. Get feature names
    # Polynomial feature names
    poly_features = preprocess.named_transformers_['poly'].get_feature_names_out(numeric_poly)
    
    # Scaled numeric features
    numeric_features = numeric_other
    
    # OneHot categorical features
    cat_features = preprocess.named_transformers_['cat'].get_feature_names_out(categorical)
    
    # Combine all
    all_features = np.concatenate([poly_features, numeric_features, cat_features])
    
    # 4. Get feature importances
    if hasattr(model_clone, 'feature_importances_'):
        importances = model_clone.feature_importances_
    else:
        raise ValueError(f"{model_name} does not have feature_importances_ attribute")
    
    # 5. Plot
    fi_df = pd.DataFrame({"feature": all_features, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(30)  # top 20 features
    
    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature",  hue="feature", data=fi_df, palette="viridis", legend=False)
    plt.title(f"Top 20 Feature Importances — {model_name}")
    plt.tight_layout()
    plt.show()

# Call it - "Random Forest"   # or "XGBoost"
#plot_feature_importance("XGBoost" , models_all, numeric_poly, numeric_other, categorical, X_train, X_test, y_train)
#plot_feature_importance("Random Forest" , numeric_poly, numeric_other, categorical, X_train, X_test, y_train)


