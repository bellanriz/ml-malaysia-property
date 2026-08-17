import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('models/best_model.pkl')

# Load cleaned data to get location/property type options
df = pd.read_csv('data/kl_property_cleaned.csv')

st.title("🏠 Malaysia Property Price Prediction")
st.write("Predict property prices in Kuala Lumpur")

# User inputs
location = st.selectbox("Location", sorted(df['Location'].unique()))
size = st.number_input("Size (sq.ft)", min_value=200, max_value=10000, value=1000)
rooms = st.slider("Rooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
car_parks = st.slider("Car Parks", 0, 5, 2)
property_type = st.selectbox("Property Type", sorted(df['Property Type'].unique()))
furnishing = st.selectbox("Furnishing", sorted(df['Furnishing'].unique()))

# Encode inputs (same as training)
location_encoded = df[df['Location'] == location]['Location_encoded'].iloc[0]
type_encoded = df[df['Property Type'] == property_type]['Property Type_encoded'].iloc[0]
furnishing_encoded = df[df['Furnishing'] == furnishing]['Furnishing_encoded'].iloc[0]

# Predict
if st.button("Predict Price"):
    features = pd.DataFrame({
        'Location_encoded': [location_encoded],
        'Rooms': [rooms],
        'Bathrooms': [bathrooms],
        'Car Parks': [car_parks],
        'Property Type_encoded': [type_encoded],
        'Size': [size],
        'Furnishing_encoded': [furnishing_encoded],
    })

    prediction_log = model.predict(features)
    price = np.expm1(prediction_log[0])

    st.success(f"Predicted Price: RM {price:,.0f}")
