import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error


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


def train_gradient_boosting(X_train, y_train, learning_rate=0.05, n_estimators=500, max_depth=4, random_state=42):
    print('8. train model : gradient_boosting')
    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
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

# ---------------------------------------------------------
# Custom RMSLE Function
# ---------------------------------------------------------
def rmsle(y_true, y_pred):
    #y_pred = np.maximum(0, y_pred)  # RMSLE requires non-negative predictions
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# ======================================================
# 1. FEATURE ENGINEERING
# ======================================================
def add_derived_features(df):

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract useful parts (but NOT using hour/day raw later)
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df['month'] = df['datetime'].dt.month

    # ----------------------------
    # Cyclical Hour Encoding
    # ----------------------------
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # ----------------------------
    # Peak Hour Flag
    # ----------------------------
    peak_hours = [7, 8, 9, 16, 17, 18, 19]
    df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)

    # ----------------------------
    # Interaction: Working day × Peak hour
    # ----------------------------
    df['working_peak'] = df['workingday'] * df['is_peak_hour']

    # ----------------------------
    # Temperature buckets
    # ----------------------------
    df['temp_bucket'] = pd.cut(
        df['temp'],
        bins=[-1, 10, 22, 30, 50],
        labels=["cold", "mild", "warm", "hot"]
    )

    # ----------------------------
    # Weather × Season interaction
    # ----------------------------
    df['weather_season'] = df['weather'].astype(str) + "_" + df['season'].astype(str)

    # ----------------------------
    # Non-linear interaction
    # ----------------------------
    df['temp_humidity'] = df['temp'] * df['humidity']

    return df


# ======================================================
# 2. PREPROCESSING PIPELINE
# ======================================================
def preprocess_data(df):
    print('2. Preprocess data...')

    print(' Before : ', list(df.columns))
    df = add_derived_features(df)
   
    # Extract target
    #y = df["count"]

    # LOG TRANSFORM TARGET
    y_log = np.log1p(df['count'])

    # Remove leakage & correlations
    df = df.drop(columns=["count", "casual", "registered", "atemp"])  
    # Drop datetime (no use)
    df = df.drop(columns=["datetime", 'hour'])

    print(' After : ', list(df.columns))
    
    # Copy dataframe
    X = df.copy()

    # Categorical features kept SMALL (no hour, no day)
    categorical_features = [
        'season', 'weather',
        'temp_bucket', 'weather_season'
    ]

    numeric_features = [
    'temp', 'humidity', 'windspeed',
    'hour_sin', 'hour_cos',
    'temp_humidity',
    'month', 'weekday',
    'is_peak_hour', 'working_peak'
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
        remainder='passthrough'
    )

    print(' Original features : ', len(df.columns))
    # Fit-transform X
    X_processed = preprocessor.fit_transform(X)
    print(' After transformation : ', X_processed.shape[1])
    print(' X_processed shape : ', X_processed.shape)

    # Verify scaling
    print(' Feature Scaling Verification:')
    print(f'   Mean of scaled features (should be ~0): {np.mean(X_processed[:, :len(numeric_features)]):.6f}')
    print(f'   Std of scaled features (should be ~1): {np.std(X_processed[:, :len(numeric_features)]):.6f}')
    print(f'   Binary features range: [{np.min(X_processed[:, len(numeric_features):]):.1f}, {np.max(X_processed[:, len(numeric_features):]):.1f}]')

    return X_processed, y_log, preprocessor

# ---------------------------------------------------------
# 5. Main execution on training dataset
# ---------------------------------------------------------
# read training data set
print('1. Reading training data...')
df = pd.read_csv("bike_train.csv")
#df.head(5)

# Preprocess data
X_processed, y_log, encoder = preprocess_data(df)

# Train/test split : 80-20 
print('3. Split train-test data...')
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_processed, y_log, test_size=0.2, random_state=42
)

# Train the models
lin_model = train_linear_regression(X_train, y_train_log)
ridge_model = train_ridge(X_train, y_train_log, alpha=1.0)
lasso_model = train_lasso(X_train, y_train_log, alpha=0.01)

rf_model = train_random_forest(X_train, y_train_log, n_estimators=400, max_depth=18)
gb_model = train_gradient_boosting(X_train, y_train_log, learning_rate=0.05, n_estimators=600, max_depth=2)

print('9. Evaluate models ... ')
results = {
    "Linear Regression": evaluate_model(lin_model, X_test, y_test_log),
    "Ridge Regression": evaluate_model(ridge_model, X_test, y_test_log),
    "Lasso Regression": evaluate_model(lasso_model, X_test, y_test_log),
    "Random Forest": evaluate_model(rf_model, X_test, y_test_log),
    "Gradient Boosting": evaluate_model(gb_model, X_test, y_test_log),
}

# Print results
print(pd.DataFrame(results).T)



