from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# Create the Flask app
app = Flask(__name__)
CORS(app)

# Load model and data once when server starts
model = joblib.load('models/best_model.pkl')
df = pd.read_csv('data/kl_property_cleaned.csv')

print(f"Model loaded: {type(model).__name__}")
print(f"Data loaded: {len(df)} rows")


@app.route('/predict', methods=['POST'])
def predict():
    """Receive property details, return predicted price."""
    data = request.json

    # Get encoded values from the dataset
    location_row = df[df['Location'] == data['location']]
    type_row = df[df['Property Type'] == data['property_type']]
    furnishing_row = df[df['Furnishing'] == data['furnishing']]

    if location_row.empty or type_row.empty or furnishing_row.empty:
        return jsonify({'error': 'Invalid input values'}), 400

    location_encoded = int(location_row['Location_encoded'].iloc[0])
    type_encoded = int(type_row['Property Type_encoded'].iloc[0])
    furnishing_encoded = int(furnishing_row['Furnishing_encoded'].iloc[0])

    # Prepare features in same order as training
    features = pd.DataFrame({
        'Location_encoded': [location_encoded],
        'Rooms': [data['rooms']],
        'Bathrooms': [data['bathrooms']],
        'Car Parks': [data['car_parks']],
        'Property Type_encoded': [type_encoded],
        'Size': [data['size']],
        'Furnishing_encoded': [furnishing_encoded],
    })

    # Predict (model returns log price, convert back)
    prediction_log = model.predict(features)
    price = float(np.expm1(prediction_log[0]))

    return jsonify({
        'predicted_price': round(price, 0),
        'price_per_sqft': round(price / data['size'], 0),
        'monthly_payment': round(price * 0.004, 0),
        'down_payment': round(price * 0.1, 0),
    })


@app.route('/options', methods=['GET'])
def get_options():
    """Return dropdown options for the frontend."""
    return jsonify({
        'locations': sorted(df['Location'].unique().tolist()),
        'property_types': sorted(df['Property Type'].unique().tolist()),
        'furnishings': sorted(df['Furnishing'].unique().tolist()),
    })


if __name__ == '__main__':
    print("API running at http://localhost:5000")
    app.run(debug=True, port=5000)

