
# ------------------------------------------------------------------
# Load test data and apply identical transformations
# ------------------------------------------------------------------
print('10. Start predicting ... ')
test_df = pd.read_csv("bike_test.csv")

# save datetime
datetime_backup = test_df["datetime"]

test_df = add_derived_features(test_df, False)

debugDate = pd.DataFrame({
    "datetime": datetime_backup,
    "day": test_df["day"],
    "month": test_df["month"],
    "year": test_df["year"],
    "hour": test_df["hour"],
    "weekday": test_df["weekday"],
})
debugDate.to_csv("datetime_debug.csv", index=False)

# Remove leakage & correlations
test_df = test_df.drop(columns=["datetime", "hour", "atemp"])

#print(' Test - Feature Engineering : ', list(test_df.columns))
print(' Test - Feature Engineering # : ', len(test_df.columns))
X_final = test_df.copy()
X_final_processed = preprocessor.transform(X_final)
print(' Test Transformation - Shape : ', X_final_processed.shape)
print(' Test Transformation - Features # : ', X_final_processed.shape[1])
#print(' Test Transformation - Features : ', feature_names)

# ------------------------------------------------------------------
# Predict using Best Model 
# ------------------------------------------------------------------
test_pred_log = cat_model.predict(X_final_processed)
test_pred = np.expm1(test_pred_log)  # reverse log1p
# No negative predictions
test_pred = np.maximum(test_pred, 0)

# ------------------------------------------------------------------
# Create submission CSV
# ------------------------------------------------------------------
submission = pd.DataFrame({
    "datetime": datetime_backup,
    "count_predicted": test_pred.round().astype(int)
})
submission.to_csv("submission.csv", index=False)
print("submission.csv generated successfully!")












from sklearn.model_selection import TimeSeriesSplit
import numpy as np

feature_names = [
    'temp', 'humidity', 'windspeed', 'hour_sin', 'hour_cos',
    'season_1', 'season_2', 'season_3', 'season_4',
    'weather_1', 'weather_2', 'weather_3', 'weather_4',
    'holiday', 'workingday', 'weekday', 'day', 'month', 'year'
]

X_processed_df = pd.DataFrame(X_processed, columns=feature_names)


# Suppose X_processed and Y (log-transformed) are ready
tscv = TimeSeriesSplit(n_splits=5)

# We will capture the last split as final train/test
for fold, (train_index, test_index) in enumerate(tscv.split(X_processed_df)):
    print(f"\n----- TimeSeriesSplit Fold {fold+1} -----")
    
    X_train, X_test = X_processed_df.iloc[train_index], X_processed_df.iloc[test_index]
    y_train_log, y_test_log = Y.iloc[train_index], Y.iloc[test_index]
    
    # Train your model (example: Gradient Boosting)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train_log)
    
    # Evaluate
    pred = model.predict(X_test)
    rmsle = rmsle(y_train_log, pred)
    print("RMSLE:", rmsle)

# After loop — last X_train/X_test become your final train-validation sets
print("\nFinal Train Shape:", X_train.shape)
print("Final Test Shape:", X_test.shape)









import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score

SEEDS = [0, 1, 7, 12, 42, 101, 2024]

def evaluate_seed(seed, pipeline, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred = np.maximum(0, y_pred)

    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rmsle, r2


# ---------------------------
# RUN MULTIPLE SEED TESTS
# ---------------------------
results = []

for seed in SEEDS:
    rmsle, r2 = evaluate_seed(seed, pipeline, X, y)
    results.append([seed, rmsle, r2])
    print(f"Seed {seed}: RMSLE={rmsle:.5f}, R²={r2:.5f}")

# Convert to DataFrame
df_results = pd.DataFrame(results, columns=["seed", "rmsle", "r2"])

print("\n------ SUMMARY ------")
print(df_results)

print("\nAverage RMSLE :", df_results["rmsle"].mean())
print("Std Dev RMSLE:", df_results["rmsle"].std())

print("\nAverage R²    :", df_results["r2"].mean())
print("Std Dev R²    :", df_results["r2"].std())
