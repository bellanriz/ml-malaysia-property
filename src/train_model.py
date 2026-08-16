import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# 1. LOAD AND PREPARE DATA (same as preprocessing)
# ============================================================
df = pd.read_csv('data/kl_property_cleaned.csv')
print(f"Loaded cleaned data: {df.shape}")

feature_cols = ['Location_encoded', 'Rooms', 'Bathrooms', 'Car Parks',
                'Property Type_encoded', 'Size', 'Furnishing_encoded']

X = df[feature_cols]
y = df['Price_log']  # Use log-transformed price for better results

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (needed for Linear, Ridge, Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# ============================================================
# 2. DEFINE MODELS
# ============================================================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

# ============================================================
# 3. TRAIN AND EVALUATE EACH MODEL
# ============================================================
print("\n" + "=" * 70)
print(f"{'Model':<25} {'R² Score':<12} {'MAE (RM)':<15} {'RMSE (RM)':<15}")
print("=" * 70)

results = {}
for name, model in models.items():
    # Tree-based models don't need scaling
    if name in ['Random Forest', 'Gradient Boosting']:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"{name:<25} {r2:<12.4f} {mae:<15,.0f} {rmse:<15,.0f}")

print("=" * 70)

# ============================================================
# 4. FIND THE BEST MODEL
# ============================================================
best_model_name = max(results, key=lambda k: results[k]['R2'])
print(f"\nBest model: {best_model_name} (R2 = {results[best_model_name]['R2']:.4f})")


rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print(f"\nFeature Importance (Random Forest):")
print("-" * 40)
for feature, importance in importances.items():
    bar = "#" * int(importance * 50)
    print(f"  {feature:<25} {importance:.4f} {bar}")

# ============================================================
# 6. SAVE THE BEST MODEL
# ============================================================
import joblib
import os

os.makedirs('models', exist_ok=True)

# Save the best performing model
best_model = models[best_model_name]
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print(f"\nModel saved to models/best_model.pkl")
