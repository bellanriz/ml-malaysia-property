import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load ('models/scaler.pkl')

print("Model loaded succesfully!")
print (f"Model type: {type(model).__name__}")

# ============================================================
# 2. DEFINE A NEW PROPERTY TO PREDICT
#    The features must be in the same order as training:
#    [Location_encoded, Rooms, Bathrooms, Car Parks,
#     Property Type_encoded, Size, Furnishing_encoded]
#
#    To know the encoded values, check kl_property_cleaned.csv
#    or use the label encoders. For now, use numbers directly.
# ============================================================

# Example: A condo in Mont Kiara
# Location_encoded = check your data (just pick a number from the dataset)
# Rooms = 3
# Bathrooms = 2
# Car Parks = 2
# Property Type_encoded = check your data
# Size = 1200 sq.ft
# Furnishing_encoded = check your data

new_property = pd.DataFrame({
    'Location_encoded': [25],
    'Rooms': [3],
    'Bathrooms': [2],
    'Car Parks': [2],
    'Property Type_encoded': [5],
    'Size': [1200],
    'Furnishing_encoded': [1],
})

# ============================================================
# 3. MAKE THE PREDICTION
#    Remember: our model predicts log(price), so we need to
#    convert back using np.expm1() (inverse of np.log1p)
# ============================================================
prediction_log = model.predict(new_property)
prediction_price = np.expm1(prediction_log[0])

print(f"\n--- Prediction ---")
print(f"Property: 3 bed, 2 bath, 1200 sqft")
print(f"Predicted Price: RM {prediction_price:,.0f}")

# ============================================================
# 4. TRY MULTIPLE PROPERTIES AT ONCE
# ============================================================
properties = pd.DataFrame({
    'Location_encoded': [25, 10, 40],
    'Rooms': [3, 2, 4],
    'Bathrooms': [2, 1, 3],
    'Car Parks': [2, 1, 3],
    'Property Type_encoded': [5, 5, 3],
    'Size': [1200, 800, 2500],
    'Furnishing_encoded': [1, 2, 0],
})

predictions_log = model.predict(properties)
predictions_price = np.expm1(predictions_log)

print(f"\n--- Batch Predictions ---")
for i, price in enumerate(predictions_price):
    print(f"  Property {i+1}: RM {price:,.0f}")