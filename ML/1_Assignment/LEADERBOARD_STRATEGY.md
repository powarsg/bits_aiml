# Leaderboard Performance Strategy Guide

## 📊 Understanding Your Current Results

### Your Local Scores:
- **Linear**: RMSLE=0.6099, R2=0.4830
- **Ridge**: RMSLE=0.6099, R2=0.4854
- **Lasso**: RMSLE=0.6485, R2=0.5845
- **RandomForest**: RMSLE=0.3169, R2=0.9445 ⭐

### Analysis:

1. **RandomForest is clearly best** (RMSLE 0.3169)
2. **High R2 (0.9445) is concerning** - might indicate overfitting
3. **Linear models are underperforming** - need better features or regularization

---

## ⚠️ **Overfitting Detection**

### Signs of Overfitting:
- **Train RMSLE << Test RMSLE** (train much better)
- **Train R2 >> Test R2** (train R2 close to 1.0)
- **Large gap between train and test performance**

### Your RandomForest:
- R2 = 0.9445 is **very high** - potential overfitting
- Need to check: Train RMSLE vs Test RMSLE gap

### How to Check:
The enhanced `fit_and_eval()` function now shows:
- Train RMSLE vs Test RMSLE
- Train R2 vs Test R2
- Gap between them

**If gap > 0.05**: Overfitting detected!

---

## 🎯 **Cross-Validation: Better Leaderboard Estimate**

### Why Cross-Validation?
- Single train/test split can be **misleading**
- CV gives **more reliable estimate** of leaderboard performance
- Uses all data for both training and validation

### What to Expect:
- **CV RMSLE** is typically **closer to leaderboard** than single split
- **CV mean** = your expected leaderboard score
- **CV std** = uncertainty (lower is better)

### Interpretation:
```
CV RMSLE: 0.35 (+/- 0.02)
→ Expected leaderboard: ~0.35
→ Range: 0.33 - 0.37 (with 95% confidence)
```

---

## 📈 **Strategies to Improve Leaderboard Performance**

### 1. **Reduce Overfitting (RandomForest)**

#### Current Settings (might be too complex):
```python
n_estimators=800, max_depth=18, min_samples_split=4, min_samples_leaf=2
```

#### Try More Regularization:
```python
RandomForestRegressor(
    n_estimators=300,      # Reduce from 800
    max_depth=12,          # Reduce from 18
    min_samples_split=10,  # Increase from 4
    min_samples_leaf=5,    # Increase from 2
    max_features='sqrt',   # Limit features per split
    random_state=42,
    n_jobs=-1
)
```

**Expected Impact**: RMSLE might increase slightly (0.32 → 0.34) but **better generalization**

---

### 2. **Ensemble Methods (Best Strategy!)**

#### A. Stacking (Recommended)
Combine multiple models:
```python
from sklearn.ensemble import StackingRegressor

# Base models
base_models = [
    ('ridge', pipe_ridge),
    ('lasso', pipe_lasso),
    ('rf', pipe_rf)
]

# Meta-model (final predictor)
stacked = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=10.0),
    cv=5
)
```

**Expected Impact**: RMSLE improvement of **-0.02 to -0.05**

#### B. Blending (Simple Ensemble)
Average predictions from multiple models:
```python
pred_rf = pipe_rf.predict(X_test)
pred_ridge = pipe_ridge.predict(X_test)
pred_blend = 0.7 * pred_rf + 0.3 * pred_ridge  # Weighted average
```

**Expected Impact**: RMSLE improvement of **-0.01 to -0.03**

---

### 3. **Feature Selection**

With 50+ features, some might be noise:

```python
# Use RandomForest feature importance
importances = pipe_rf.named_steps["model"].feature_importances_
feature_importance = pd.Series(importances, index=feature_names)
top_features = feature_importance.nlargest(30).index  # Keep top 30

# Retrain with only top features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]
```

**Expected Impact**: RMSLE improvement of **-0.01 to -0.03** (reduces overfitting)

---

### 4. **Hyperparameter Tuning**

Use GridSearchCV to find best parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [200, 300, 500],
    'model__max_depth': [10, 12, 15],
    'model__min_samples_split': [5, 10, 15],
    'model__min_samples_leaf': [2, 5, 10]
}

grid_search = GridSearchCV(
    pipe_rf, param_grid, cv=5,
    scoring='neg_mean_squared_log_error',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
```

**Expected Impact**: RMSLE improvement of **-0.02 to -0.05**

---

### 5. **Advanced Models**

#### A. XGBoost (Often Better Than RandomForest)
```python
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Expected Impact**: RMSLE improvement of **-0.03 to -0.08**

#### B. LightGBM (Fast and Accurate)
```python
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
```

**Expected Impact**: RMSLE improvement of **-0.03 to -0.08**

#### C. CatBoost (Handles Categorical Well)
```python
from catboost import CatBoostRegressor

catboost = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=False
)
```

**Expected Impact**: RMSLE improvement of **-0.03 to -0.08**

---

## 🎯 **Expected Leaderboard Performance**

### Based on Your Local Scores:

| Model | Local RMSLE | Expected Leaderboard | Confidence |
|-------|-------------|---------------------|------------|
| RandomForest | 0.3169 | **0.32 - 0.38** | Medium (if overfitting) |
| Ridge | 0.6099 | 0.60 - 0.65 | High |
| Lasso | 0.6485 | 0.64 - 0.70 | High |
| Linear | 0.6099 | 0.60 - 0.65 | High |

### If Overfitting Detected:
- RandomForest leaderboard: **0.35 - 0.42** (worse than local)
- Need to reduce complexity

### If Good Generalization:
- RandomForest leaderboard: **0.30 - 0.35** (similar to local)
- Can try more complex models

---

## 📋 **Action Plan**

### Step 1: Check Overfitting
Run the code and check:
- Train RMSLE vs Test RMSLE gap
- If gap > 0.05 → Reduce RandomForest complexity

### Step 2: Run Cross-Validation
- Check CV RMSLE scores
- CV mean = expected leaderboard score
- Use CV to select best model

### Step 3: Reduce Overfitting (if needed)
- Reduce `max_depth` to 10-12
- Increase `min_samples_split` to 10
- Increase `min_samples_leaf` to 5
- Retrain and check CV again

### Step 4: Try Ensemble
- Stack RandomForest + Ridge + Lasso
- Or blend predictions
- Check CV improvement

### Step 5: Try Advanced Models
- XGBoost or LightGBM
- Often better than RandomForest
- Check CV performance

### Step 6: Final Submission
- Use model with **best CV score**
- Not necessarily best local test score
- CV is more reliable for leaderboard

---

## 🏆 **Target Leaderboard Scores**

### Competitive Ranges:
- **Top 10%**: RMSLE < 0.30
- **Top 25%**: RMSLE < 0.35
- **Top 50%**: RMSLE < 0.40
- **Average**: RMSLE 0.40 - 0.50

### Your Goal:
- Current: 0.3169 (local)
- Target: < 0.35 (leaderboard)
- Stretch: < 0.30 (top 10%)

---

## 💡 **Key Insights**

1. **CV > Single Split**: Cross-validation is more reliable
2. **Overfitting is Dangerous**: High local score doesn't mean good leaderboard
3. **Ensemble Works**: Combining models often improves performance
4. **Feature Selection Helps**: Remove noisy features
5. **Regularization Matters**: Don't make models too complex

---

## 🔍 **Red Flags**

### ⚠️ Watch Out For:
- Train RMSLE << Test RMSLE (overfitting)
- CV RMSLE >> Test RMSLE (unlucky split)
- CV std > 0.05 (unstable model)
- R2 > 0.95 (likely overfitting)

### ✅ Good Signs:
- Train RMSLE ≈ Test RMSLE (good generalization)
- CV RMSLE ≈ Test RMSLE (reliable estimate)
- CV std < 0.03 (stable model)
- R2 0.85 - 0.95 (reasonable fit)

---

## 📊 **Monitoring Checklist**

Before submitting:
- [ ] Check train vs test gap (< 0.05)
- [ ] Run 5-fold CV
- [ ] Compare CV mean to test score
- [ ] Check feature importance
- [ ] Try ensemble if single model overfits
- [ ] Final CV score < 0.35 (target)

---

*Use this guide to systematically improve your leaderboard performance!*

