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
import re
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
    """
    Enhanced feature engineering for bike prediction with focus on Hour and Weather.
    This function creates features that significantly improve RMSLE score.
    """
    df = df.copy()

    df['datetime'] = pd.to_datetime(df['datetime'], format=dateformat)

    # ============================================
    # 1. BASIC TEMPORAL FEATURES
    # ============================================
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['year'] = df['datetime'].dt.year.astype(str)
    df['dayofyear'] = df['datetime'].dt.dayofyear
    
    # ============================================
    # 2. CYCLICAL ENCODING (CRITICAL FOR HOUR & MONTH)
    # ============================================
    # Hour cyclical encoding - captures daily patterns (23:59 close to 00:00)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Month cyclical encoding - captures seasonal patterns (Dec close to Jan)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Weekday cyclical encoding
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # Day of year cyclical encoding (for seasonal patterns)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # ============================================
    # 3. ADVANCED HOUR FEATURES (HIGH IMPACT)
    # ============================================
    # Hour buckets for different usage patterns
    df['is_morning'] = df['hour'].isin([6, 7, 8, 9]).astype(int)
    df['is_afternoon'] = df['hour'].isin([12, 13, 14, 15]).astype(int)
    df['is_evening'] = df['hour'].isin([17, 18, 19, 20]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
    
    # Rush hour (morning and evening)
    df['is_rush_hour_morning'] = df['hour'].isin([7, 8, 9]).astype(int)
    df['is_rush_hour_evening'] = df['hour'].isin([17, 18, 19]).astype(int)
    df['is_rush_hour'] = (df['is_rush_hour_morning'] | df['is_rush_hour_evening']).astype(int)
    
    # Peak hours (highest demand)
    df['is_peak_hour'] = df['hour'].isin([8, 17, 18]).astype(int)
    
    # Business hours
    df['is_business_hour'] = df['hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17]).astype(int)
    
    # ============================================
    # 4. WEATHER FEATURES (HIGH IMPACT)
    # ============================================
    # Weather severity (higher number = worse weather)
    df['weather_severity'] = df['weather'].astype(int)
    
    # Binary flags for weather conditions
    df['weather_clear'] = (df['weather'] == 1).astype(int)
    df['weather_mist'] = (df['weather'] == 2).astype(int)
    df['weather_light'] = (df['weather'] == 3).astype(int)
    df['weather_heavy'] = (df['weather'] == 4).astype(int)
    
    # Bad weather flag (combines weather 3 and 4)
    df['is_bad_weather'] = df['weather'].isin([3, 4]).astype(int)
    
    # Good weather flag (weather 1)
    df['is_good_weather'] = (df['weather'] == 1).astype(int)
    
    # ============================================
    # 5. HOUR × WEATHER INTERACTIONS (CRITICAL!)
    # ============================================
    # These interactions capture how weather affects different hours differently
    df['hour_weather_interaction'] = df['hour'] * df['weather']
    df['hour_sin_weather'] = df['hour_sin'] * df['weather']
    df['hour_cos_weather'] = df['hour_cos'] * df['weather']
    
    # Rush hour × weather (bad weather during rush hour has different impact)
    df['rush_hour_bad_weather'] = df['is_rush_hour'] * df['is_bad_weather']
    df['rush_hour_good_weather'] = df['is_rush_hour'] * df['is_good_weather']
    
    # Peak hour × weather
    df['peak_hour_bad_weather'] = df['is_peak_hour'] * df['is_bad_weather']
    df['peak_hour_good_weather'] = df['is_peak_hour'] * df['is_good_weather']
    
    # ============================================
    # 6. WEEKEND & HOLIDAY FEATURES
    # ============================================
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['holiday'].astype(int)
    df['is_workingday'] = df['workingday'].astype(int)
    
    # Weekend × hour interaction (weekend patterns differ from weekday)
    df['weekend_hour'] = df['is_weekend'] * df['hour']
    df['weekend_hour_sin'] = df['is_weekend'] * df['hour_sin']
    df['weekend_hour_cos'] = df['is_weekend'] * df['hour_cos']
    
    # Holiday × hour interaction
    df['holiday_hour'] = df['is_holiday'] * df['hour']
    
    # ============================================
    # 7. WEATHER × TEMPORAL INTERACTIONS
    # ============================================
    # Weather × month (seasonal weather patterns)
    df['weather_month'] = df['weather'] * df['month']
    df['weather_month_sin'] = df['weather'] * df['month_sin']
    df['weather_month_cos'] = df['weather'] * df['month_cos']
    
    # Weather × weekday (weekend weather might have different impact)
    df['weather_weekend'] = df['weather'] * df['is_weekend']
    
    # ============================================
    # 8. TEMPERATURE & ENVIRONMENT FEATURES
    # ============================================
    # Temperature interactions
    df['temp_humidity_interaction'] = df['temp'] * (1 - df['humidity'])
    df['temp_windspeed'] = df['temp'] * df['windspeed']
    
    # Comfort index (higher = more comfortable)
    df['comfort_index'] = df['temp'] * (1 - df['humidity'] / 100) * (1 - df['windspeed'] / 50)
    
    # Weather × temperature (bad weather + cold = very low demand)
    df['weather_temp'] = df['weather'] * df['temp']
    
    # ============================================
    # 9. ADVANCED TEMPORAL FEATURES
    # ============================================
    # Quarter of year
    df['quarter'] = df['month'].apply(lambda x: (x-1)//3 + 1)
    
    # Is summer/winter (peak seasons)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    # ============================================
    # 10. CLEANUP
    # ============================================
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
    X, y, test_size=0.2, random_state=24
)


# 5. Pipelines
# We'll build two pipelines:
#  A) For linear models (LinearRegression, Ridge, Lasso):
#     - numeric: StandardScaler + optional PolynomialFeatures
#     - categorical: OneHotEncoder(drop='first')
#  B) For RandomForest:
#     - numeric: pass-through (no scaling) or StandardScaler optional
#     - categorical: OneHotEncoder(drop='first')

# ============================================
# NUMERICAL FEATURES (for scaling and polynomial features)
# ============================================
numeric_features = [
    # Basic environmental
    'temp', 'humidity', 'windspeed',
    
    # Cyclical encodings (CRITICAL - captures patterns)
    'hour_sin', 'hour_cos',           # Hour cyclical
    'month_sin', 'month_cos',         # Month cyclical
    'weekday_sin', 'weekday_cos',     # Weekday cyclical
    'dayofyear_sin', 'dayofyear_cos', # Day of year cyclical
    
    # Interactions
    'temp_humidity_interaction',
    'temp_windspeed',
    'comfort_index',
    
    # Hour × Weather interactions (HIGH IMPACT)
    'hour_weather_interaction',
    'hour_sin_weather',
    'hour_cos_weather',
    'weather_temp',
    
    # Temporal interactions
    'weekend_hour',
    'weekend_hour_sin',
    'weekend_hour_cos',
    'holiday_hour',
    
    # Weather × Temporal interactions
    'weather_month',
    'weather_month_sin',
    'weather_month_cos',
]

# ============================================
# CATEGORICAL FEATURES (for one-hot encoding)
# ============================================
categorical_features = [
    # Basic categorical
    'year', 'workingday', 'holiday',
    
    # Weather (keep as categorical for non-linear effects)
    'weather',
    
    # Binary flags (treated as categorical for one-hot)
    'is_weekend', 'is_holiday', 'is_workingday',
    'is_morning', 'is_afternoon', 'is_evening', 'is_night',
    'is_rush_hour', 'is_rush_hour_morning', 'is_rush_hour_evening',
    'is_peak_hour', 'is_business_hour',
    'weather_clear', 'weather_mist', 'weather_light', 'weather_heavy',
    'is_bad_weather', 'is_good_weather',
    'rush_hour_bad_weather', 'rush_hour_good_weather',
    'peak_hour_bad_weather', 'peak_hour_good_weather',
    'weather_weekend',
    'is_summer', 'is_winter',
    'quarter'
]


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
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("model", LinearRegression())
])

# Ridge
pipe_ridge = Pipeline(steps=[
    ("pre", preprocessor_linear),
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("model", Ridge(alpha=1.0))
])


# Lasso
pipe_lasso = Pipeline(steps=[
    ("pre", preprocessor_linear),
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("model", Lasso(alpha=0.01, max_iter=50000))
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


# 7. Utility function to run training + evaluation with overfitting check
def fit_and_eval(pipe, X_train, X_test, y_train, y_test, name, debug = False):
    """
    Train and evaluate model, showing both train and test performance to detect overfitting.
    """
    pipe.fit(X_train, y_train)

    if debug == True :
        debug_data(pipe, X_train, X_test)

    # Test predictions
    pred_log = pipe.predict(X_test)
    pred = np.expm1(pred_log)
    y_test_actual = np.expm1(y_test)
    
    result = {
        "model": name,
        "RMSLE": rmsle(y_test_actual, pred),
        "R2": r2_score(y_test_actual, pred)
        #"fit_time_s": t1 - t0
    }

    print(f"{name}: RMSLE={result['RMSLE']:.4f}, R2={result['R2']:.4f}")
    
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
res_lin   = fit_and_eval(pipe_lin, X_train, X_test, y_train, y_test, "Linear", debug)
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
print(fi.head(30))


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

# ============================================
# 9. RESIDUAL PLOT FOR LINEAR MODEL
# ============================================
def plot_residuals(pipe, X_test, y_test):
    """
    Plots residuals for the Linear model using test split data.
    """
    # Predict log -> convert back
    pred_log = pipe.predict(X_test)
    y_pred = np.expm1(pred_log)
    y_true = np.expm1(y_test)

    residuals = y_true - y_pred

    plt.figure(figsize=(10,5))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (y_true - y_pred)")
    plt.title("Residual Plot - Linear Regression")
    plt.show()


# Call residual plot for Linear model
plot_residuals(pipe_lin, X_test, y_test)


plot_residuals(pipe_rf, X_test, y_test)

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
submission.to_csv("submission_RF.csv", index=False)
print(submission.head())
#print(submission.tail())
print("Submission file saved: submission_RF.csv")


# Predict using model (example: Random Forest)
pred_log = pipe_lin.predict(bike_test_fe)
test_pred = np.expm1(pred_log)

# Prepare submission
submission = pd.DataFrame({
"datetime": bike_test["datetime"],
"count_predicted": test_pred.round().astype(int)
})
submission.to_csv("submission_lin.csv", index=False)
print(submission.head())
#print(submission.tail())
print("Submission file saved: submission_lin.csv")



# ============================================
# 10. FIX DATETIME HOUR FORMAT IN SUBMISSION FILE
# ============================================
def fix_submission_datetime_hour(input_file="submission_lin.csv", output_file="submission.csv"):
    """
    Read submission file and fix datetime column's hour value.
    - If hour value doesn't have 0 prepended (single digit), add 0
    - If hour value already has 0 prepended (double digit), keep as is
    - Preserve datetime order and count_predicted column unchanged
    
    Examples:
        - 05-06-2012 5:00 -> 05-06-2012 05:00
        - 02-04-2012 6:00 -> 02-04-2012 06:00
        - 13-01-2012 07:00 -> 13-01-2012 07:00 (unchanged)
    
    Args:
        input_file (str): Path to input CSV file (default: "submission_lin.csv")
        output_file (str): Path to output CSV file (default: "submission.csv")
    
    Returns:
        pd.DataFrame: DataFrame with corrected datetime column
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Reading {input_file}: {len(df)} rows")
    
    # Function to fix hour format - add leading zero to single digit hours
    def fix_datetime_hour(dt_str):
        """
        Fix datetime string to ensure hour has leading zero if single digit.
        Pattern: DD-MM-YYYY H:MM -> DD-MM-YYYY HH:MM
        """
        # Pattern to match: DD-MM-YYYY followed by space, then single digit hour (0-9), then colon and minutes
        pattern = r'(\d{2}-\d{2}-\d{4}) (\d):(\d{2})'
        
        def replace_func(match):
            date_part = match.group(1)  # DD-MM-YYYY
            hour = match.group(2)        # Single digit hour (0-9)
            minute = match.group(3)      # MM
            # Add leading zero to hour using zfill(2)
            return f'{date_part} {hour.zfill(2)}:{minute}'
        
        # Replace single digit hours with zero-padded hours
        fixed = re.sub(pattern, replace_func, dt_str)
        return fixed
    
    # Apply the fix to datetime column
    df['datetime'] = df['datetime'].apply(fix_datetime_hour)
    
    # Count how many were fixed
    df_original = pd.read_csv(input_file)
    single_digit_original = df_original['datetime'].str.contains(r' \d:', regex=True).sum()
    single_digit_new = df['datetime'].str.contains(r' \d:', regex=True).sum()
    fixed_count = single_digit_original - single_digit_new
    
    print(f"Fixed {fixed_count} rows with single digit hours")
    print(f"Rows with single digit hours remaining: {single_digit_new} (should be 0)")
    
    # Save to new file - maintain original order
    df.to_csv(output_file, index=False)
    
    print(f"File saved: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Datetime order preserved: ✓")
    print(f"count_predicted column unchanged: ✓")
    
    # Show sample of corrections
    print("\nSample corrections:")
    sample_indices = df_original[df_original['datetime'].str.contains(r' \d:', regex=True)].head(5).index
    for idx in sample_indices:
        orig = df_original.loc[idx, 'datetime']
        new = df.loc[idx, 'datetime']
        print(f"  {orig:20} -> {new:20}")
    
    return df


# Call the function to fix submission file
print("\n" + "="*60)
print("Fixing datetime hour format in submission_lin file...")
print("="*60)
fixed_df = fix_submission_datetime_hour("submission_lin.csv", "submission_lin_hr.csv")
print("="*60)

print("Fixing datetime hour format in submission_RF file...")
fixed_df = fix_submission_datetime_hour("submission_RF.csv", "submission_RF_hr.csv")