import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error


# ---------------------------------------------------------
# 1. Custom RMSLE Function
# ---------------------------------------------------------
def rmsle(y_true, y_pred):
    #y_pred = np.maximum(0, y_pred)  # RMSLE requires non-negative predictions
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def add_derived_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    df['weekday'] = df['datetime'].dt.weekday
    #df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    #df['temp_atemp_diff'] = df['temp'] - df['atemp']
    #df['humidity_windspeed_ratio'] = df['humidity'] / (df['windspeed'] + 1e-3)

    df = df.drop(columns=['datetime'])
    return df


# ---------------------------------------------------------
# 2. Preprocessing data
# ---------------------------------------------------------
def preprocess_data(df):
    print('2. preprocess_data ...')

    y = df['count']

    print(f'  Before : {list(df.columns)}')
   
    df = add_derived_features(df)

    # Remove dependent fields
    df = df.drop(columns=['count', 'casual', 'registered'])

    # Remove redundant field
    df = df.drop(columns=['atemp'])

    print(f'  After : {list(df.columns)}')

    X = df.copy()

    # Numeric features
    numeric_features = ['temp', 'humidity', 'windspeed']
    numeric_transformer = StandardScaler()

    # Categorical (One-Hot)
    categorical_features = ['season', 'weather', 'hour', 'month', 'weekday']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Build full transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    print("  Fitting & transforming X ...")
    X_transformed = preprocessor.fit_transform(X)

    # convert sparse → dense
    #if hasattr(X_transformed, "toarray"):
    #    X_transformed = X_transformed.toarray()

    print("  X transformed shape:", X_transformed.shape)

    return X_transformed, y, preprocessor



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
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred)  # RMSLE requires non-negative predictions

    results = {
        "RMSLE-Custom": rmsle(y_test, y_pred),
        "RMSLE": np.sqrt(mean_squared_log_error(y_test, y_pred)),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    return results


# ---------------------------------------------------------
# 5. Main execution on training dataset
# ---------------------------------------------------------
# read training data set
print('1. Reading training data...')
df = pd.read_csv("bike_train.csv")
df.head(5)

# Preprocess data
X_processed, y, encoder = preprocess_data(df)

# Train/test split : 80-20 
print('3. Split train-test data...')
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Train the models
lin_model = train_linear_regression(X_train, y_train)
ridge_model = train_ridge(X_train, y_train, alpha=1.0)
lasso_model = train_lasso(X_train, y_train, alpha=0.01)
rf_model = train_random_forest(X_train, y_train, n_estimators=400, max_depth=18)
gb_model = train_gradient_boosting(X_train, y_train, learning_rate=0.05, n_estimators=600, max_depth=4)

print('9. Evaluate models ... ')
results = {
    "Linear Regression": evaluate_model(lin_model, X_test, y_test),
    "Ridge Regression": evaluate_model(ridge_model, X_test, y_test),
    "Lasso Regression": evaluate_model(lasso_model, X_test, y_test),
    "Random Forest": evaluate_model(rf_model, X_test, y_test),
    "Gradient Boosting": evaluate_model(gb_model, X_test, y_test),
}

# Print results
print(pd.DataFrame(results).T)



