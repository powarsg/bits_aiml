# Bike Sharing: Feature Engineering + Modeling
# Jupyter-ready Python script (can be pasted into a notebook cell-by-cell)
# Steps: Load -> Clean -> Feature engineering -> Split -> Pipelines -> Models (Linear, Ridge, Lasso, RandomForest)

# 0. Install / imports (run in a cell)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score

# Helper metric functions
import math

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    # clip predictions to avoid negative values inside log
    y_pred_clip = np.clip(y_pred, 0, None)
    return math.sqrt(mean_squared_log_error(y_true, y_pred_clip))

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

def feature_engineer(df, dateformat, drop_original_datetime=True, ):
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], format=dateformat)

    # basic parts
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year.astype(str)  # treat year as categorical (string) or numeric depending on model

    # binary features
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7,8,9,16,17,18,19]).astype(int)

    # interactions (small and interpretable)
    # temp * (1 - humidity) might capture comfortable conditions; keep simple interaction
    df['temp_humidity_interaction'] = df['temp'] * (1 - df['humidity'])

    # drop redundant cols if present
    if 'atemp' in df.columns:
        # choose to keep 'temp' and drop 'atemp'
        df = df.drop(columns=['atemp'])
    if 'season' in df.columns:
        df = df.drop(columns=['season'])

    if drop_original_datetime:
        df = df.drop(columns=['datetime'])

    return df

# 4. Prepare features and target, then split
# Example usage (uncomment after loading df):
# df = feature_engineer(df)
# target = 'count'
# X = df.drop(columns=[target, 'casual', 'registered'] if set(['casual','registered']).issubset(df.columns) else [target])
# y = df[target]

# Then split:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Pipelines
# We'll build two pipelines:
#  A) For linear models (LinearRegression, Ridge, Lasso):
#     - numeric: StandardScaler + optional PolynomialFeatures
#     - categorical: OneHotEncoder(drop='first')
#  B) For RandomForest:
#     - numeric: pass-through (no scaling) or StandardScaler optional
#     - categorical: OneHotEncoder(drop='first')

numeric_features = ['temp', 'humidity', 'windspeed', 'hour', 'month', 'is_weekend', 'is_rush_hour', 'temp_humidity_interaction']
# Note: 'year' kept as categorical below, 'workingday', 'holiday', 'weather' are categorical
categorical_features = ['year', 'workingday', 'holiday', 'weather']

# transformers
numeric_transformer_linear = Pipeline(steps=[
    ('scaler', StandardScaler()),
    # Polynomial will be injected inside model pipeline with degree=2 when required
])

numeric_transformer_rf = Pipeline(steps=[
    # for RF we do not need scaling; keep as passthrough or add simple scaler if desired
])

categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

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
    ('pre', preprocessor_linear),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # degree can be changed to 2 for interactions
    ('model', LinearRegression())
])

# Ridge
pipe_ridge = Pipeline(steps=[
    ('pre', preprocessor_linear),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge())
])

# Lasso
pipe_lasso = Pipeline(steps=[
    ('pre', preprocessor_linear),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Lasso(max_iter=5000))
])

# Random Forest
pipe_rf = Pipeline(steps=[
    ('pre', preprocessor_rf),
    ('model', RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1))
])

# 7. Utility function to run training + evaluation
from time import time

def fit_and_eval(pipeline, X_train, X_test, y_train, y_test, name='model'):
    t0 = time()
    pipeline.fit(X_train, y_train)
    t1 = time()
    preds = pipeline.predict(X_test)
    res = {
        'name': name,
        #'RMSE': rmse(y_test, preds),
        'RMSLE': rmsle(y_test, preds),
        'R2': r2_score(y_test, preds),
        'fit_time_s': t1 - t0
    }
    print(f"{name} -> RMSLE: {res['RMSLE']:.4f}, R2: {res['R2']:.4f}, fit_time: {res['fit_time_s']:.1f}s")
    return res

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

# 9. Full example workflow (uncomment and run):
# ---------------------------------------------------------------------
df = pd.read_csv('bike_train.csv', parse_dates=['datetime'])
df = feature_engineer(df, dateformat_train)
target = 'count'
# # drop casual/registered if present
if set(['casual','registered']).issubset(df.columns):
    df = df.drop(columns=['casual','registered'])

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Baseline runs
res_lin = fit_and_eval(pipe_lin, X_train, X_test, y_train, y_test, name='Linear')
res_ridge = fit_and_eval(pipe_ridge, X_train, X_test, y_train, y_test, name='Ridge')
res_lasso = fit_and_eval(pipe_lasso, X_train, X_test, y_train, y_test, name='Lasso')
res_rf = fit_and_eval(pipe_rf, X_train, X_test, y_train, y_test, name='RandomForest')

# # If you want polynomial degree=2 for linear models (be careful):
#pipe_ridge.set_params(poly__degree=2)
#res_ridge_poly2 = fit_and_eval(pipe_ridge, X_train, X_test, y_train, y_test, name='Ridge_poly2')

# 10. Feature importance for RandomForest (after fitting pipe_rf)
# if you run pipe_rf.fit(X_train, y_train):
#   # get feature names after column transformer
pre = pipe_rf.named_steps['pre']
#   # numeric feature names (passed through)
num_features = numeric_features
#   # categorical feature names from OneHotEncoder
ohe = pre.named_transformers_['cat']
cat_cols = list(ohe.get_feature_names_out(categorical_features))
feature_names = num_features + cat_cols
importances = pipe_rf.named_steps['model'].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(fi.head(30))

# ---------------------------------------------------------------------
# End of script
# Notes & tips:
# - Trees (RandomForest) usually perform much better than plain linear models on this dataset.
# - Use RMSLE as the main metric since count is a non-negative skewed target.
# - Use polynomial features sparingly. degree=2 may help linear models but can explode dims when you OHE categories.
# - If you want to include 'weekday', prefer engineered binary flags (is_weekend, is_rush_hour) rather than raw weekday.
# - Consider using Gradient Boosting (XGBoost / LightGBM / CatBoost) after this pipeline for best performance.



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
test_pred = pipe_rf.predict(bike_test_fe)


# Prepare submission
submission = pd.DataFrame({
"datetime": bike_test["datetime"],
"count_predicted": test_pred.round().astype(int)
})

print(submission.head())
print(submission.tail())

submission.to_csv("submission_RF_28Nov.csv", index=False)
print("Submission file saved: bike_submission.csv")
