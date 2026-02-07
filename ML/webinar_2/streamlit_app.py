import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("End-to-End Regression Demo with Regularization")

st.write("""
Interactively explore **Linear, Lasso, and Ridge Regression**.
Adjust hyperparameters for Lasso and Ridge to see the effect.
""")

# Load pipelines
lr_pipeline = joblib.load('lr_pipeline.pkl')
lasso_pipeline = joblib.load('lasso_pipeline.pkl')
ridge_pipeline = joblib.load('ridge_pipeline.pkl')

# Sidebar: hyperparameter sliders
st.sidebar.header("Adjust Hyperparameters")
alpha_lasso = st.sidebar.slider("Lasso alpha", 0.0, 5.0, 0.5, 0.1)
alpha_ridge = st.sidebar.slider("Ridge alpha", 0.0, 5.0, 1.0, 0.1)

# Input for custom data
st.subheader("Enter Feature Values (comma separated, 5 features)")
input_data = st.text_input("Example: 0.5, -1.2, 2.3, 0, 1.5")
if input_data:
    try:
        X_input = np.array([float(x) for x in input_data.split(',')]).reshape(1,-1)
        # Update Lasso and Ridge with new alpha
        lasso_pipeline.set_params(regressor__alpha=alpha_lasso)
        ridge_pipeline.set_params(regressor__alpha=alpha_ridge)

        # Fit again (on synthetic data for demo purposes)
        # In real scenario, use GridSearchCV / trained pipelines
        # Here we keep synthetic dataset inside pipelines
        # Predict
        y_lr = lr_pipeline.predict(X_input)[0]
        y_lasso = lasso_pipeline.predict(X_input)[0]
        y_ridge = ridge_pipeline.predict(X_input)[0]

        st.write(f"**Predictions:**")
        st.write(f"Linear Regression: {y_lr:.3f}")
        st.write(f"Lasso Regression: {y_lasso:.3f}")
        st.write(f"Ridge Regression: {y_ridge:.3f}")

    except:
        st.error("Please enter exactly 5 numeric values separated by commas.")

# Compare coefficients
st.subheader("Coefficient Comparison")
coefs_df = pd.DataFrame({
    "Linear": lr_pipeline.named_steps['regressor'].coef_,
    "Lasso": lasso_pipeline.named_steps['regressor'].coef_,
    "Ridge": ridge_pipeline.named_steps['regressor'].coef_
})
st.dataframe(coefs_df)

# Plot coefficients
plt.figure(figsize=(10,5))
plt.plot(coefs_df["Linear"], 'o-', label='Linear')
plt.plot(coefs_df["Lasso"], 's-', label='Lasso')
plt.plot(coefs_df["Ridge"], 'x-', label='Ridge')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Effect of Regularization")
plt.legend()
st.pyplot(plt)
