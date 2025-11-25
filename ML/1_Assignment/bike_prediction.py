import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

# ---------------------------------------------------------
# Custom RMSLE Function
# ---------------------------------------------------------
def rmsle(y_true, y_pred):
    #y_pred = np.maximum(0, y_pred)  # RMSLE requires non-negative predictions
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def parse_datetime(x):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M"):
        try:
            return pd.to_datetime(x, format=fmt)
        except:
            pass
    return pd.to_datetime(x)   # fallback

def parse_train_datetime(x):
    return pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")

def parse_test_datetime(x):
    return pd.to_datetime(x, format="%d-%m-%Y %H:%M")

# ------------------------------------------------------------------
# Load training data
# ------------------------------------------------------------------
# read training data set
print('1. Reading training data...')
df = pd.read_csv("bike_train.csv")
#df.head(5)

# LOG TRANSFORM TARGET
Y = np.log1p(df['count'])

#Y = df['count']

# ------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------
def add_derived_features(df, isTrain):

    # Parse datetime
    #df["datetime"] = pd.to_datetime(df["datetime"])
    #df["datetime"] = df["datetime"].apply(parse_datetime)

    if isTrain == True :
        df["datetime"] = df["datetime"].apply(parse_train_datetime)
    else:
        df["datetime"] = df["datetime"].apply(parse_test_datetime)

    # Extract useful parts (but NOT using hour/year raw later)
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    # ----------------------------
    # Cyclical Hour Encoding
    # ----------------------------
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # ----------------------------
    # Peak Hour Flag
    # ----------------------------
    peak_hours = [7, 8, 9, 16, 17, 18, 19]
    #df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)

    # ----------------------------
    # Interaction: Working day × Peak hour = 0,1
    # ----------------------------
    #df['is_working_peak'] = df['workingday'] * df['hour'].isin(peak_hours).astype(int)

    # ----------------------------
    # Non-linear interaction
    # ----------------------------
    #df['temp_humidity'] = df['temp'] * df['humidity']
    
    #df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    # ----------------------------
    # Temperature buckets
    # ----------------------------
    #df['temp_bucket'] = pd.cut(
    #    df['temp'],
    #    bins=[-1, 10, 22, 30, 50],
    #    labels=["cold", "mild", "warm", "hot"]
    #)
    # ----------------------------
    # Weather × Season interaction
    # ----------------------------
    #df['weather_season'] = df['weather'].astype(str) + "_" + df['season'].astype(str)

    return df


print('2. Preprocess data...')
print(f' Original Shape : {df.shape}' )
print(f' Original Columns : {len(df.columns)}' )
print(' Before : ', list(df.columns))

df = add_derived_features(df, True)

# Remove leakage & correlations
df = df.drop(columns=["count", "casual", "registered"])  
# Drop datetime (no use)
df = df.drop(columns=["datetime", "hour", "atemp"])

print(' After - Feature Engineering : ', list(df.columns))
print(' After - Feature Engineering # : ', len(df.columns))


# ------------------------------------------------------------------
# Feature sets
# ------------------------------------------------------------------
numeric_features = [
    "temp", "humidity", "windspeed",
    "hour_sin", "hour_cos",
   # "temp_humidity"
]
categorical_features = [
    "season", "weather", 
    "holiday", "workingday",
    "weekday", "day", "month", "year"
    #"is_working_peak"
]
#all_features = categorical_features + numeric_features
#X = df[all_features]
X = df.copy()

# ------------------------------------------------------------------
# Transform features
# ------------------------------------------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough',
    force_int_remainder_cols=False
)
def get_feature_names(preprocessor):
    output_features = []
    for name, transformer, cols in preprocessor.transformers_:
        #print(f'name = {name}')
        #if name == "remainder":
        #   continue
        if hasattr(transformer, "get_feature_names_out"):
            ft_names = transformer.get_feature_names_out(cols)
        else:
            ft_names = cols
        
        #print(f'ft_names = {ft_names}')
        output_features.extend(ft_names)
    return output_features


# Fit transform
X_processed = preprocessor.fit_transform(X)
# Preprocess data
#X_processed, y_log, encoder = preprocess_data(df)
feature_names = get_feature_names(preprocessor)
print(' After Transformation - Shape : ', X_processed.shape)
print(' After Transformation - Features # : ', X_processed.shape[1])
print(' All Features : ', feature_names)


# Train-Test data split : 80-20 
print('3. Split train-test data...')
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_processed, Y, test_size=0.20, random_state=42
)

# ---------------------------------------------------------
# Model functions
# ---------------------------------------------------------
def train_linear_regression(X_train, y_train):
    print('4. train model : linear_regression')
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge(X_train, y_train, alpha=1.0):
    print('5. train model : ridge')
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso(X_train, y_train, alpha=0.001):
    print('6. train model : lasso')
    model = Lasso(alpha=alpha, max_iter=20000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=300, max_depth=None, random_state=42):
    print('7. train model : random_forest')
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, learning_rate=0.05, n_estimators=500, max_depth=4, min_samples_leaf=1, min_samples_split=2, subsample=1.0, random_state=42):
    print('8. train model : gradient_boosting')
    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        subsample=subsample,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

from catboost import CatBoostRegressor
def train_catboost(X_train, y_train, 
                   iterations=1500, 
                   learning_rate=0.03, 
                   depth=8,
                   random_state=42):
    print('8.1. train model : CatBoost')
    
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function='RMSE',
        random_seed=random_state,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    return model

from lightgbm import LGBMRegressor
def train_lightgbm(X_train, y_train,
                   n_estimators=1500,
                   learning_rate=0.03,
                   max_depth=-1,
                   num_leaves=64,
                   random_state=42):
    print('8.2 train model : LightGBM')
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        objective='regression',
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model

# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------
def evaluate_model(model, X_test, y_test_log):
    # Convert y_test back to original count scale
    y_test = np.expm1(y_test_log)

    # Predict log(count)
    y_pred_log = model.predict(X_test)
    # Convert prediction back
    y_pred = np.expm1(y_pred_log)

    # Safety for RMSLE
    y_pred = np.maximum(0, y_pred)
    y_test = np.maximum(0, y_test)

    #y_pred = model.predict(X_test)
    #y_pred = np.maximum(0, y_pred)  # RMSLE requires non-negative predictions

    results = {
        "RMSLE": rmsle(y_test, y_pred),
        "RMSLE-Sklearn": np.sqrt(mean_squared_log_error(y_test, y_pred)),
        #"RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        #"MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    return results


# Train the models
lin_model = train_linear_regression(X_train, y_train_log)
#ridge_model = train_ridge(X_train, y_train_log, alpha=1.0)
#lasso_model = train_lasso(X_train, y_train_log, alpha=0.01)
rf_model = train_random_forest(X_train, y_train_log, n_estimators=800, max_depth=17)
#gb_model = train_gradient_boosting(X_train, y_train_log, learning_rate=0.0309, n_estimators=862, max_depth=5)
# hyper parameter - tunned
gb_tuned = train_gradient_boosting( X_train, y_train_log, learning_rate=0.0309, n_estimators=862, max_depth=5, min_samples_leaf=3, min_samples_split=7, subsample=0.8147)

# Example training call
cat_model = train_catboost(X_train, y_train_log, iterations=1800, learning_rate=0.03, depth=8)

# Example training call
#lgb_model = train_lightgbm(X_train, y_train_log, n_estimators=1800, learning_rate=0.03, num_leaves=70)



print('9. Evaluate models ... ')
results = {
    "Linear Regression": evaluate_model(lin_model, X_test, y_test_log),
    #"Ridge Regression": evaluate_model(ridge_model, X_test, y_test_log),
    #"Lasso Regression": evaluate_model(lasso_model, X_test, y_test_log),
    "Random Forest": evaluate_model(rf_model, X_test, y_test_log),
    "Gradient Boosting": evaluate_model(gb_tuned, X_test, y_test_log),
    "CatBoost" : evaluate_model(cat_model, X_test, y_test_log)
    #"LightGBM": evaluate_model(lgb_model, X_test, y_test_log)
}

# Print results
print(pd.DataFrame(results).T)




import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(name, model, feature_names):
    # --- Get importances ---
    importances = model.feature_importances_
    
    # --- Sort by importance ---
    sorted_idx = np.argsort(importances)

    # --- Plot ---
    plt.figure(figsize=(10, 14))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title(name + " - Feature Importance", fontsize=16)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Plot Residual for model
# ---------------------------------------------------------
def plot_model_residual(name, model, X_test, y_test_log):
    # Convert y_test back to original count scale
    y_test = np.expm1(y_test_log)

    # 1. Predict log(count)
    y_pred_log = model.predict(X_test)
    # Convert prediction back
    y_pred = np.expm1(y_pred_log)

    # Safety for RMSLE
    y_pred = np.maximum(0, y_pred)
    y_test = np.maximum(0, y_test)

    # 2. Compute residuals
    residuals = y_test - y_pred
    
    # 3. Plot residuals
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot - "+name)
    plt.show()

# Call it
#plot_feature_importance("Gradient Boosting", gb_tuned, feature_names)
#plot_feature_importance("Cat Boosting", cat_model, feature_names)

# plot residuals
#plot_model_residual('Gradient Boosting ', gb_tuned, X_test, y_test_log)
#plot_model_residual('Cat Boost ', cat_model, X_test, y_test_log)



# ------------------------------------------------------------------
# Load test data and apply identical transformations
# ------------------------------------------------------------------
print('10. Start predicting ... ')
test_df = pd.read_csv("bike_test.csv")

# save datetime
datetime_backup = test_df["datetime"]

test_df = add_derived_features(test_df, False)

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
# Predict using CB Model 
# ------------------------------------------------------------------
test_pred_log = cat_model.predict(X_final_processed)
test_pred = np.expm1(test_pred_log)  # reverse log1p
# No negative predictions
test_pred = np.maximum(test_pred, 0)
submission = pd.DataFrame({
    "datetime": datetime_backup,
    "count_predicted": test_pred.round().astype(int)
})
submission.to_csv("submission.csv", index=False)
print("submission.csv generated successfully!")

# ------------------------------------------------------------------
# Predict using GB Model 
# ------------------------------------------------------------------
test_pred_log_GB = gb_tuned.predict(X_final_processed)
test_pred = np.expm1(test_pred_log_GB)  # reverse log1p
# No negative predictions
test_pred = np.maximum(test_pred, 0)
submission = pd.DataFrame({
    "datetime": datetime_backup,
    "count_predicted": test_pred.round().astype(int)
})
submission.to_csv("submission_GB.csv", index=False)
print("submission_GB.csv generated successfully!")

# ------------------------------------------------------------------
# Predict using RF Model 
# ------------------------------------------------------------------
test_pred_log_RF = rf_model.predict(X_final_processed)
test_pred = np.expm1(test_pred_log_RF)  # reverse log1p
# No negative predictions
test_pred = np.maximum(test_pred, 0)
submission = pd.DataFrame({
    "datetime": datetime_backup,
    "count_predicted": test_pred.round().astype(int)
})
submission.to_csv("submission_RF.csv", index=False)
print("submission_RF.csv generated successfully!")