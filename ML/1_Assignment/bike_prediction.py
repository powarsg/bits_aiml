import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

import math
from time import time


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))


# 1. Load your dataset (replace path with your csv)
# df = pd.read_csv('train.csv', parse_dates=['datetime'])
# For notebook: uncomment above and set correct path

# 2. Basic cleaning and insights
# - If dataset already has 'count' and separated 'casual'/'registered', we'll predict 'count'
# - Drop duplicates / missing handling as needed

# 3. Feature engineering (clean & important features only)
# We'll follow the "recommended" recipe:
# Keep: hour, month, year, is_weekend, is_rush_hour, temp (drop atemp), humidity, windspeed, weather, workingday, holiday
# Drop: season (redundant with month), atemp (redundant with temp)
dateformat_train = "%Y-%m-%d %H:%M:%S"
dateformat_test = "%d-%m-%Y %H:%M"

def feature_engineer(df, dateformat, drop_dt=True):
    df = df.copy()

    df['datetime'] = pd.to_datetime(df['datetime'], format=dateformat)

    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year.astype(str)

    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    df['temp_humidity_interaction'] = df['temp'] * (1 - df['humidity'])

    if 'atemp' in df.columns:
        df = df.drop(columns=['atemp'])
    if 'season' in df.columns:
        df = df.drop(columns=['season'])

    if drop_dt:
        df = df.drop(columns=['datetime'])

    return df

# 4. Prepare features and target, then split
df = pd.read_csv("bike_train.csv")

df = feature_engineer(df, dateformat_train)

target = "count"

# Drop casual + registered if present
if set(['casual', 'registered']).issubset(df.columns):
    df = df.drop(columns=['casual', 'registered'])

X = df.drop(columns=[target])

# ✔ CRITICAL FIX: LOG TARGET
y = np.log1p(df[target])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Pipelines
# We'll build two pipelines:
#  A) For linear models (LinearRegression, Ridge, Lasso):
#     - numeric: StandardScaler + optional PolynomialFeatures
#     - categorical: OneHotEncoder(drop='first')
#  B) For RandomForest:
#     - numeric: pass-through (no scaling) or StandardScaler optional
#     - categorical: OneHotEncoder(drop='first')

numeric_features = [
    'temp', 'humidity', 'windspeed',
    'hour', 'month', 'is_weekend',
    'is_rush_hour', 'temp_humidity_interaction'
]

categorical_features = ['year', 'workingday', 'holiday', 'weather']


# transformers
numeric_transformer_linear = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

preprocessor_linear = ColumnTransformer(transformers=[
    ('num', numeric_transformer_linear, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

preprocessor_rf = ColumnTransformer(transformers=[
    ('num', 'passthrough', numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# 6. Model pipelines
# Linear
pipe_lin = Pipeline(steps=[
    ("pre", preprocessor_linear),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

# Ridge
pipe_ridge = Pipeline(steps=[
    ("pre", preprocessor_linear),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", Ridge(alpha=1.0))
])


# Lasso
pipe_lasso = Pipeline(steps=[
    ("pre", preprocessor_linear),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", Lasso(alpha=0.001, max_iter=50000))
])


# Random Forest
pipe_rf = Pipeline(steps=[
    ("pre", preprocessor_rf),
    ("model", RandomForestRegressor(
        n_estimators=800, max_depth=18,
        min_samples_split=4, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))
])


# 7. Utility function to run training + evaluation
def fit_and_eval(pipe, X_train, X_test, y_train, y_test, name, debug = False):
    t0 = time()
    pipe.fit(X_train, y_train)
    t1 = time()

    if debug == True :
        debug_data(pipe, X_train, X_test)

    pred_log = pipe.predict(X_test)
    pred = np.expm1(pred_log)
    y_test_actual = np.expm1(y_test)

    result = {
        "model": name,
        "RMSLE": rmsle(y_test_actual, pred),
        "R2": r2_score(y_test_actual, pred),
        "fit_time_s": t1 - t0
    }

    print(f"{name}: RMSLE={result['RMSLE']:.4f}, R2={result['R2']:.4f}, Time={result['fit_time_s']:.2f}s")
    return result

def debug_data(pipe, X_train, X_test):
    # ----------------------------
    # 1. Preprocess only
    # ----------------------------
    print("Before preprocessing:")
    print("  X_train :", X_train.shape)
    print("  X_test :", X_test.shape)

    X_train_final = pipe.named_steps["pre"].fit_transform(X_train)
    X_test_final = pipe.named_steps["pre"].transform(X_test)

    print("After preprocessing:")
    print("  X_train_pre :", X_train_final.shape)
    print("  X_test_pre :", X_test_final.shape)

    # ----------------------------
    # 2. Apply Polynomial Features
    # ----------------------------
    if "poly" in pipe.named_steps:
        X_train_final = pipe.named_steps["poly"].fit_transform(X_train_final)
        X_test_final  = pipe.named_steps["poly"].transform(X_test_final)

        print("After polynomial:")
        print("  X_train_poly:", X_train_final.shape)
        print("  X_test_poly:", X_test_final.shape)

    # ----------------------------
    # 3. Convert to DataFrame
    # ----------------------------
    feature_names = pipe.named_steps["pre"].get_feature_names_out()

    if "poly" in pipe.named_steps:
        feature_names = pipe.named_steps["poly"].get_feature_names_out(feature_names)

    df_train_final = pd.DataFrame(X_test_final, columns=feature_names)
    print(df_train_final.shape)
    print(df_train_final.head())
    print(df_train_final.tail())



debug = True
# # Baseline runs
res_lin   = fit_and_eval(pipe_lin, X_train, X_test, y_train, y_test, "Linear")
res_ridge = fit_and_eval(pipe_ridge, X_train, X_test, y_train, y_test, "Ridge")
res_lasso = fit_and_eval(pipe_lasso, X_train, X_test, y_train, y_test, "Lasso")
res_rf    = fit_and_eval(pipe_rf, X_train, X_test, y_train, y_test, "RandomForest")


# Feature Importance (Random Forest)
pipe_rf.fit(X_train, y_train)

pre = pipe_rf.named_steps["pre"]
ohe = pre.named_transformers_["cat"]

cat_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_names

importances = pipe_rf.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(fi.head(20))


# 8. Hyperparameter tuning suggestions (small grid examples)
# For polynomial features with linear models, try degree 1 (baseline) and 2 (interactions).
# For Ridge/Lasso tune alpha. For RF tune n_estimators, max_depth.

# Example GridSearch for Ridge with polynomial degree 2
# WARNING: Polynomial degree=2 blows up features. Use with caution (monitor feature count).

# grid = {
#     'poly__degree': [1, 2],
#     'model__alpha': [0.01, 0.1, 1, 10]
# }
# gs = GridSearchCV(pipe_ridge, grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# gs.fit(X_train, y_train)
# print(gs.best_params_, math.sqrt(-gs.best_score_))

# RandomForest grid example
# rf_grid = {
#     'model__n_estimators': [100, 200],
#     'model__max_depth': [None, 10, 20]
# }
# gs_rf = GridSearchCV(pipe_rf, rf_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
# gs_rf.fit(X_train, y_train)
# print(gs_rf.best_params_)



# Load test data
bike_test = pd.read_csv("bike_test.csv")

# Apply same feature engineering function
bike_test_fe = feature_engineer(bike_test, dateformat_test)

# Ensure train and test have same columns
missing_cols = set(X_train.columns) - set(bike_test_fe.columns)

print(f' Missing columns : {missing_cols}')
for col in missing_cols:
    bike_test_fe[col] = 0

bike_test_fe = bike_test_fe[X_train.columns]

# Predict using best model (example: Random Forest)
pred_log = pipe_rf.predict(bike_test_fe)
test_pred = np.expm1(pred_log)

# Prepare submission
submission = pd.DataFrame({
"datetime": bike_test["datetime"],
"count_predicted": test_pred.round().astype(int)
})
submission.to_csv("submission_RF_Log.csv", index=False)
print(submission.head())
print(submission.tail())
print("Submission file saved: submission_RF_Log.csv")


# Predict using model (example: Random Forest)
pred_log = pipe_lin.predict(bike_test_fe)
test_pred = np.expm1(pred_log)

# Prepare submission
submission = pd.DataFrame({
"datetime": bike_test["datetime"],
"count_predicted": test_pred.round().astype(int)
})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(submission.tail())
print("Submission file saved: submission.csv")