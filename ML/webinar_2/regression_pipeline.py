import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import joblib

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(100, 5)
true_coef = np.array([3, 0, 2, 0, 1.5])

y = X @ true_coef + np.random.randn(100) * 0.5

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numeric features
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, [0,1,2,3,4])
])

# Pipeline for Linear Regression
lr_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

# Pipeline for Lasso
lasso_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', Lasso(alpha=0.5))
])
lasso_pipeline.fit(X_train, y_train)

# Pipeline for Ridge
ridge_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])
ridge_pipeline.fit(X_train, y_train)

# Save pipelines
joblib.dump(lr_pipeline, 'lr_pipeline.pkl')
joblib.dump(lasso_pipeline, 'lasso_pipeline.pkl')
joblib.dump(ridge_pipeline, 'ridge_pipeline.pkl')

print("Pipelines saved successfully!")
