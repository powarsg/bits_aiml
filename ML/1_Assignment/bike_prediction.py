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

    # Base components
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    # Cyclic Encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Binary features
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # Interaction features
    df["temp_humidity"] = df["temp"] * df["humidity"]
    df["feels_like_diff"] = df["atemp"] - df["temp"]

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
X = train_df[FEATURES]

# -----------------------------------------------------------
# STEP 3: TRAIN/TEST SPLIT FOR VALIDATION
# -----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
     #, random_state=42
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
        n_estimators=800, max_depth=18,
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
y_pred = pipeline.predict(X_test)
y_pred = np.maximum(0, y_pred)

rmsle = mean_squared_log_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

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
submission.to_csv("submission_RF_py.csv", index=False)
print(f"submission_RF_py.csv generated successfully!")
# -----------------------------------------------------------
# DONE
# -----------------------------------------------------------

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
plot_feature_importance("Random Forest" , numeric_poly, numeric_other, categorical, X_train, X_test, y_train)