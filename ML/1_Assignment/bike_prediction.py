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
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m-%d-%Y %H:%M", "%d-%m-%Y %H:%M"):
        try:
            return pd.to_datetime(x, format=fmt)
        except:
            pass
    return pd.to_datetime(x)   # fallback

# ------------------------------------------------------------------
# 1. Load training data
# ------------------------------------------------------------------
# read training data set
print('1. Reading training data...')
df = pd.read_csv("bike_train.csv")
#df.head(5)
# LOG TRANSFORM TARGET
Y = np.log1p(df['count'])

# ======================================================
# 2. FEATURE ENGINEERING
# ======================================================
def add_derived_features(df):

    # Parse datetime
    #df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].apply(parse_datetime)

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
    df['is_working_peak'] = df['workingday'] * df['hour'].isin(peak_hours).astype(int)

    # ----------------------------
    # Non-linear interaction
    # ----------------------------
    df['temp_humidity'] = df['temp'] * df['humidity']
    
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
print(f' Original Features : {len(df.columns)-1}' )
print(' Before : ', list(df.columns))

df = add_derived_features(df)

# Remove leakage & correlations
df = df.drop(columns=["count", "casual", "registered"])  
# Drop datetime (no use)
df = df.drop(columns=["datetime", "hour", "atemp"])

print(' After - Feature Engineering : ', list(df.columns))
print(' After - Feature Engineering # : ', len(df.columns))


# ------------------------------------------------------------------
# 3. Feature sets
# ------------------------------------------------------------------
categorical_features = ["season", "weather"]
numeric_features = [
    "temp", "humidity", "windspeed",
    "hour_sin", "hour_cos",
    "temp_humidity",
    "month", "weekday", "day", 
    #'holiday', 'workingday', 'year',
    #"is_peak_hour", 
    "is_working_peak"
]
#all_features = categorical_features + numeric_features
#X = df[all_features]
X = df.copy()

# ------------------------------------------------------------------
# 4. Transform features
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
print(' After Transformation - Features : ', feature_names)


# Train-Test data split : 80-20 
print('3. Split train-test data...')
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_processed, Y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 3. Model functions
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

# ---------------------------------------------------------
# 4. Evaluation model
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
        "RMSLE-Custom": rmsle(y_test, y_pred),
        "RMSLE-Sklearn": np.sqrt(mean_squared_log_error(y_test, y_pred)),
        #"RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        #"MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    return results


# Train the models
lin_model = train_linear_regression(X_train, y_train_log)
ridge_model = train_ridge(X_train, y_train_log, alpha=1.0)
lasso_model = train_lasso(X_train, y_train_log, alpha=0.01)
rf_model = train_random_forest(X_train, y_train_log, n_estimators=800, max_depth=17)
#gb_model = train_gradient_boosting(X_train, y_train_log, learning_rate=0.0309, n_estimators=862, max_depth=5)

# hyper parameter - tunned
gb_tuned = train_gradient_boosting( X_train, y_train_log, learning_rate=0.0309, n_estimators=862, max_depth=5, min_samples_leaf=3, min_samples_split=7, subsample=0.8147)

print('9. Evaluate models ... ')
results = {
    "Linear Regression": evaluate_model(lin_model, X_test, y_test_log),
    #"Ridge Regression": evaluate_model(ridge_model, X_test, y_test_log),
    #"Lasso Regression": evaluate_model(lasso_model, X_test, y_test_log),
    "Random Forest": evaluate_model(rf_model, X_test, y_test_log),
    "Gradient Boosting": evaluate_model(gb_tuned, X_test, y_test_log),
}

# Print results
print(pd.DataFrame(results).T)

# ------------------------------------------------------------------
# 6. Load test data and apply identical transformations
# ------------------------------------------------------------------
print('10. Start TESTING ... ')
test_df = pd.read_csv("bike_test.csv")

# save datetime
datetime_backup = test_df["datetime"]

test_df = add_derived_features(test_df)

# Remove leakage & correlations
test_df = test_df.drop(columns=["datetime", "hour", "atemp"])

#print(' Test - Feature Engineering : ', list(test_df.columns))
print(' Test - Feature Engineering # : ', len(test_df.columns))
X_test = test_df.copy()
X_test_processed = preprocessor.transform(X_test)
print(' Test Transformation - Shape : ', X_test_processed.shape)
print(' Test Transformation - Features # : ', X_test_processed.shape[1])
#print(' Test Transformation - Features : ', feature_names)

# ------------------------------------------------------------------
# 7. Predict using best Model 
# ------------------------------------------------------------------
test_pred_log = gb_tuned.predict(X_test_processed)
test_pred = np.expm1(test_pred_log)  # reverse log1p
# No negative predictions
test_pred = np.maximum(test_pred, 0)

# ------------------------------------------------------------------
# 8. Create submission CSV
# ------------------------------------------------------------------
submission = pd.DataFrame({
    "datetime": datetime_backup,
    "count": test_pred.round().astype(int)
})
submission.to_csv("submission.csv", index=False)


test_pred_log_rf = rf_model.predict(X_test_processed)
test_pred_rf = np.expm1(test_pred_log_rf)  # reverse log1p
# No negative predictions
test_pred_rf = np.maximum(test_pred_rf, 0)
submission = pd.DataFrame({
    "datetime": datetime_backup,
    "count_GB": test_pred.round().astype(int),
    "count_RF": test_pred_rf.round().astype(int)
})
submission.to_csv("submission_compare.csv", index=False)

print("submission.csv generated successfully!")



# ======================================================
# 2. PREPROCESSING
# ======================================================
def preprocess_data(df):
    print('2. Preprocess data...')
    print(f' Original Shape : {df.shape}' )
    print(f' Original Features : {len(df.columns)-1}' )
    print(' Before : ', list(df.columns))
    
    df = add_derived_features(df)
   
    # Extract target
    #y = df["count"]
    # LOG TRANSFORM TARGET
    y_log = np.log1p(df['count'])

    # Remove leakage & correlations
    df = df.drop(columns=["count", "casual", "registered", "atemp"])  
    # Drop datetime (no use)
    df = df.drop(columns=["datetime", "hour"])

    print(' After - Feature Engineering : ', list(df.columns))
    print(' After - Feature Engineering  # : ', len(df.columns))

    # Categorical features
    categorical_features = [
        'season', 'weather'
        #,'temp_bucket', 'weather_season'
    ]

    numeric_features = [
    'temp', 'humidity', 'windspeed',
    'hour_sin', 'hour_cos',
    'temp_humidity',
    'month', 'weekday', 'day',
     #'hour', 'is_peak_hour', 'is_night',
    'is_working_peak'
    ]

    # ----------------------------
    # Transformers
    # ----------------------------
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # ----------------------------
    # ColumnTransformer
    # ----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        force_int_remainder_cols=False
    )

    # Copy dataframe
    X = df.copy()

    # Fit-transform X
    X_processed = preprocessor.fit_transform(X)

    feature_names = get_feature_names(preprocessor)
    print(' After Transformation - Shape : ', X_processed.shape)
    print(' After Transformation - Features # : ', X_processed.shape[1])
    print(' After Transformation - Features : ', feature_names)
    
    #X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    #print(X_processed[:5])

    return X_processed, y_log, preprocessor



import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(name, model, preprocessor):
    # --- Get feature names from ColumnTransformer ---
    num_features = preprocessor.transformers_[0][2]
    cat_features = list(preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2]))
    remainder = preprocessor.transformers_[2][2] if len(preprocessor.transformers_) > 2 else []

    feature_names = num_features + cat_features + remainder

    # --- Get importances ---
    importances = model.feature_importances_
    
    # --- Sort by importance ---
    sorted_idx = np.argsort(importances)

    # --- Plot ---
    plt.figure(figsize=(10, 14))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title(name + " Feature Importance", fontsize=16)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

# Call it
plot_feature_importance( "Gradient Boosting" , gb_tuned, preprocessor)

