# Numerical vs Categorical Features for Bike Prediction (Low RMSLE Guide)

## Overview
Properly categorizing features as **numerical** or **categorical** is crucial for achieving low RMSLE scores in bike sharing prediction. This document explains the reasoning behind each choice.

---

## 📊 **NUMERICAL FEATURES** (Continuous/Ordinal Values)

### Why Numerical?
- **Continuous values**: Can take any value in a range (e.g., temperature: 20.5°C, 21.3°C)
- **Ordinal relationships**: Values have meaningful order and distance (e.g., 30°C is "twice as hot" as 15°C)
- **Mathematical operations**: Can perform addition, subtraction, multiplication (e.g., temp × humidity interaction)

### Numerical Features in Bike Prediction:

#### 1. **Temperature (`temp`)**
- **Type**: Continuous numerical
- **Why**: Temperature directly affects bike usage (warmer = more riders)
- **Processing**: StandardScaler (for linear models) or passthrough (for tree models)
- **Impact**: High - Strong correlation with bike demand

#### 2. **Humidity (`humidity`)**
- **Type**: Continuous numerical (0-100%)
- **Why**: Affects comfort level and bike usage patterns
- **Processing**: StandardScaler or passthrough
- **Impact**: Medium-High - Influences outdoor activity

#### 3. **Windspeed (`windspeed`)**
- **Type**: Continuous numerical
- **Why**: Strong winds discourage biking
- **Processing**: StandardScaler or passthrough
- **Impact**: Medium - Affects safety perception

#### 4. **Cyclical Hour Encoding (`hour_sin`, `hour_cos`)**
- **Type**: Numerical (derived from hour)
- **Why**: 
  - Hours are **cyclical** (23:59 is close to 00:00)
  - Sin/Cos encoding captures this cyclicality
  - More efficient than 24 one-hot columns
  - Preserves proximity relationships (hour 1 is close to hour 2)
- **Formula**: 
  - `hour_sin = sin(2π × hour / 24)`
  - `hour_cos = cos(2π × hour / 24)`
- **Impact**: **VERY HIGH** - Daily patterns are the strongest predictor

#### 5. **Cyclical Month Encoding (`month_sin`, `month_cos`)**
- **Type**: Numerical (derived from month)
- **Why**: 
  - Months are cyclical (December is close to January)
  - Captures seasonal patterns
  - Better than treating month as categorical
- **Formula**: 
  - `month_sin = sin(2π × month / 12)`
  - `month_cos = cos(2π × month / 12)`
- **Impact**: High - Seasonal variations in bike usage

#### 6. **Interaction Features (`temp_humidity_interaction`)**
- **Type**: Numerical (derived feature)
- **Why**: 
  - Combines two numerical features
  - Captures non-linear relationships
  - Example: High temp + high humidity = very uncomfortable
- **Formula**: `temp × (1 - humidity)` or `temp × humidity`
- **Impact**: Medium - Captures complex weather effects

---

## 🏷️ **CATEGORICAL FEATURES** (Discrete Labels/Classes)

### Why Categorical?
- **Discrete values**: Limited set of distinct categories (e.g., weather: 1, 2, 3, 4)
- **No inherent order**: Categories don't have meaningful numerical relationships
- **One-Hot Encoding**: Converts each category to a binary column (0 or 1)

### Categorical Features in Bike Prediction:

#### 1. **Year (`year`)**
- **Type**: Categorical (even though it's a number)
- **Why**: 
  - Limited values (e.g., 2011, 2012)
  - Each year may have different patterns (growth trends, policy changes)
  - No meaningful "distance" between years (2012 isn't "twice" 2011)
- **Processing**: OneHotEncoder
- **Impact**: Medium - Captures year-over-year trends

#### 2. **Weather (`weather`)**
- **Type**: Categorical (ordinal-like but treated as categorical)
- **Values**: 
  - 1 = Clear/Few clouds
  - 2 = Mist/Cloudy
  - 3 = Light Snow/Light Rain
  - 4 = Heavy Rain/Ice/Snow
- **Why**: 
  - Discrete categories with distinct effects
  - Non-linear impact (weather 4 is much worse than weather 1, but not "4x worse")
- **Processing**: OneHotEncoder
- **Impact**: **HIGH** - Weather significantly affects bike usage

#### 3. **Working Day (`workingday`)**
- **Type**: Binary categorical (0 or 1)
- **Why**: 
  - Discrete binary feature
  - Different usage patterns on workdays vs weekends
- **Processing**: OneHotEncoder (or can be kept as binary numerical)
- **Impact**: High - Strong weekday/weekend patterns

#### 4. **Holiday (`holiday`)**
- **Type**: Binary categorical (0 or 1)
- **Why**: 
  - Discrete binary feature
  - Holidays have different usage patterns
- **Processing**: OneHotEncoder
- **Impact**: Medium - Affects demand patterns

#### 5. **Is Weekend (`is_weekend`)**
- **Type**: Binary categorical (0 or 1)
- **Why**: 
  - Derived binary feature
  - Weekends have different usage patterns than weekdays
- **Processing**: OneHotEncoder
- **Impact**: High - Strong weekend effect

#### 6. **Is Rush Hour (`is_rush_hour`)**
- **Type**: Binary categorical (0 or 1)
- **Why**: 
  - Derived binary feature
  - Rush hours (7-9 AM, 4-7 PM) have peak demand
- **Processing**: OneHotEncoder
- **Impact**: **VERY HIGH** - Captures peak demand periods

---

## 🎯 **Why This Leads to Low RMSLE**

### 1. **Proper Feature Representation**
- **Numerical features** → StandardScaler ensures all features are on similar scales
- **Categorical features** → OneHotEncoder creates distinct binary columns
- **Result**: Models can learn optimal weights/patterns for each feature type

### 2. **Cyclical Encoding Benefits**
- **Hour/Month as numerical (cyclical)**: 
  - Captures that hour 23 is close to hour 0
  - More efficient (2 features vs 24 one-hot columns)
  - Better for both linear and tree models
- **Result**: Better daily/seasonal pattern capture → Lower RMSLE

### 3. **Interaction Features**
- **Numerical × Numerical**: Creates non-linear relationships
- **Example**: `temp × humidity` captures "feels like" temperature
- **Result**: Better weather effect modeling → Lower RMSLE

### 4. **Model-Specific Optimization**
- **Linear Models (Ridge, Lasso)**: 
  - Numerical features: StandardScaler + PolynomialFeatures
  - Categorical features: OneHotEncoder
- **Tree Models (RandomForest)**: 
  - Numerical features: Passthrough (trees handle scaling naturally)
  - Categorical features: OneHotEncoder
- **Result**: Each model type gets optimal preprocessing → Lower RMSLE

---

## 📋 **Current Feature Classification (bike_prediction.py)**

### Numerical Features:
```python
numeric_features = [
    'temp',                    # Continuous temperature
    'humidity',                # Continuous humidity (0-100%)
    'windspeed',               # Continuous wind speed
    'temp_humidity_interaction', # Interaction feature
    'hour_sin', 'hour_cos',    # Cyclical hour encoding
    'month_sin', 'month_cos'   # Cyclical month encoding
]
```

### Categorical Features:
```python
categorical_features = [
    'year',           # Year (2011, 2012, etc.)
    'workingday',      # Binary: 0 or 1
    'holiday',         # Binary: 0 or 1
    'weather',         # Weather category (1-4)
    'is_weekend',      # Binary: 0 or 1
    'is_rush_hour'     # Binary: 0 or 1
]
```

---

## ⚠️ **Common Mistakes to Avoid**

### ❌ **Mistake 1: Treating Hour as Categorical**
- **Problem**: Creates 24 one-hot columns, loses cyclicality
- **Fix**: Use cyclical encoding (sin/cos)
- **Impact**: Can improve RMSLE by 0.01-0.05

### ❌ **Mistake 2: Treating Month as Categorical**
- **Problem**: Loses seasonal continuity
- **Fix**: Use cyclical encoding (sin/cos)
- **Impact**: Better seasonal pattern capture

### ❌ **Mistake 3: Not Scaling Numerical Features (for linear models)**
- **Problem**: Features on different scales (temp: 0-40, humidity: 0-100)
- **Fix**: Use StandardScaler for linear models
- **Impact**: Prevents features from dominating

### ❌ **Mistake 4: Treating Weather as Numerical**
- **Problem**: Assumes linear relationship (weather 4 isn't "4x worse" than weather 1)
- **Fix**: Use OneHotEncoder (categorical)
- **Impact**: Better captures non-linear weather effects

---

## 📈 **Expected RMSLE Improvements**

| Feature Type | Without Proper Encoding | With Proper Encoding | Improvement |
|-------------|------------------------|---------------------|-------------|
| Hour (categorical) | ~0.45-0.50 | ~0.40-0.45 | -0.05 |
| Hour (cyclical) | ~0.40-0.45 | ~0.35-0.40 | -0.05 |
| Month (categorical) | ~0.42-0.47 | ~0.40-0.45 | -0.02 |
| Month (cyclical) | ~0.40-0.45 | ~0.38-0.43 | -0.02 |
| Weather (numerical) | ~0.45-0.50 | ~0.40-0.45 | -0.05 |
| Weather (categorical) | ~0.40-0.45 | ~0.38-0.43 | -0.02 |

**Combined Effect**: Proper feature encoding can improve RMSLE by **0.10-0.15** points!

---

## 🔍 **How to Verify Your Classification**

### Check if Feature Should be Numerical:
1. ✅ Can you perform mathematical operations? (add, multiply)
2. ✅ Do values have meaningful order and distance?
3. ✅ Is it continuous or has many unique values?
4. ✅ Does it represent a measurement?

### Check if Feature Should be Categorical:
1. ✅ Limited set of distinct categories?
2. ✅ No meaningful numerical relationships?
3. ✅ Each category has distinct behavior?
4. ✅ Binary or small number of classes?

---

## 💡 **Best Practices Summary**

1. **Cyclical Features** (hour, month, weekday) → **Numerical with sin/cos encoding**
2. **Continuous Measurements** (temp, humidity, windspeed) → **Numerical**
3. **Discrete Categories** (weather, year) → **Categorical with OneHotEncoder**
4. **Binary Flags** (workingday, holiday, is_weekend) → **Categorical**
5. **Interactions** (temp × humidity) → **Numerical**
6. **Scale numerical features** for linear models (StandardScaler)
7. **Passthrough numerical features** for tree models (they handle it naturally)

---

## 🎓 **Key Takeaways**

- **Numerical**: Continuous values, cyclical patterns (with encoding), interactions
- **Categorical**: Discrete categories, binary flags, limited unique values
- **Cyclical encoding** (sin/cos) is better than one-hot for hour/month
- **Proper classification** can improve RMSLE by 0.10-0.15 points
- **Model-specific**: Linear models need scaling, tree models don't

---

*This classification strategy is optimized for achieving low RMSLE scores in bike sharing prediction tasks.*

