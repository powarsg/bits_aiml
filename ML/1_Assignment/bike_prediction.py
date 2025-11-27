# ================================================================
#  BIKE SHARING FULL PIPELINE
#  TRAIN  →  SAVE MODEL  →  LOAD TEST  →  PREDICT  →  SAVE OUTPUT
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
from joblib import dump, load

dateformat_train = "%Y-%m-%d %H:%M:%S"
dateformat_test = "%d-%m-%Y %H:%M"
# ---------------------------------------------------------
# Custom RMSLE Function
# ---------------------------------------------------------
def rmsle(y_true, y_pred):
    #y_pred = np.maximum(0, y_pred)  # RMSLE requires non-negative predictions
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def parse_train_datetime(x):
    return pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")

def parse_test_datetime(x):
    return pd.to_datetime(x, format="%d-%m-%Y %H:%M")

# -----------------------------------------------------------
# STEP 1: FEATURE ENGINEERING (REUSED FOR TRAIN + TEST)
# -----------------------------------------------------------

def add_datetime_features(df, dateformat):

    df["datetime"] = pd.to_datetime(df["datetime"], format=dateformat)

    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18,19]).astype(int)

    return df

# -----------------------------------------------------------
# STEP 2: LOAD TRAIN CSV AND ENGINEER FEATURES
# -----------------------------------------------------------

TRAIN_FILE = "bike_train.csv"     # make sure this exists
TEST_FILE = "bike_test.csv"       # test data
MODEL_FILE = "bike_final_model.pkl"
OUTPUT_FILE = "submission_new_RF.csv"

train_df = pd.read_csv(TRAIN_FILE)
train_df = add_datetime_features(train_df, dateformat_train)

# Target
y = train_df["count"]

# Base Features
numeric_poly = ["temp", "atemp"]
numeric_other = ["humidity", "windspeed"]


categorical = [
    "season", "weather", "holiday", "workingday",
    "hour", "weekday", "month", "year",
    "is_weekend", "is_rush_hour"
]

FEATURES = numeric_poly + numeric_other + categorical
X = train_df[FEATURES]

# -----------------------------------------------------------
# STEP 3: TRAIN/TEST SPLIT FOR VALIDATION
# -----------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# STEP 4: PREPROCESSING PIPELINE
# -----------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False), numeric_poly),
        ("num", StandardScaler(), numeric_other),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ],
    remainder="drop"
)

# -----------------------------------------------------------
# STEP 5: MODEL — Random Forest (Best Performing)
# -----------------------------------------------------------

model = RandomForestRegressor(
        n_estimators=300, max_depth=20,
        min_samples_split=4, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# -----------------------------------------------------------
# STEP 6: TRAIN MODEL
# -----------------------------------------------------------
print("Training model...")
pipeline.fit(X_train, y_train)

# -----------------------------------------------------------
# STEP 7: VALIDATE MODEL
# -----------------------------------------------------------
y_pred = pipeline.predict(X_val)
y_pred = np.maximum(0, y_pred)

rmsle = mean_squared_log_error(y_val, y_pred) ** 0.5
r2 = r2_score(y_val, y_pred)

print("\n----- MODEL PERFORMANCE -----")
print("RMSLE :", rmsle)
print("R²    :", r2)

# -----------------------------------------------------------
# STEP 8: SAVE MODEL (PIPELINE + PREPROCESSING TOGETHER)
# -----------------------------------------------------------
dump(pipeline, MODEL_FILE)
print(f"\nModel saved as: {MODEL_FILE}")

# -----------------------------------------------------------
# STEP 9: LOAD TEST DATA AND PREDICT
# -----------------------------------------------------------

print("\nLoading test file...")
test_df = pd.read_csv(TEST_FILE)
# save datetime
datetime_backup = test_df["datetime"]

test_df = add_datetime_features(test_df, dateformat_test)

X_test = test_df[FEATURES]

pipeline_loaded = load(MODEL_FILE)

test_pred = pipeline_loaded.predict(X_test)
test_pred = np.maximum(0, test_pred)

submission = pd.DataFrame({
    "datetime": datetime_backup,
    "count_predicted": test_pred.round().astype(int)
})
submission.to_csv(OUTPUT_FILE, index=False)
print(f"{OUTPUT_FILE} generated successfully!")
