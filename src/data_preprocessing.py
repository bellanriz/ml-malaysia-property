"""
Step 2: Data Preprocessing
- Clean Price, Size, Rooms columns (extract numbers from text)
- Handle missing values
- Encode categorical variables
- Split into train/test sets
"""
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/kl_property.csv')
print(f"Original shape: {df.shape}")

# 1. CLEAN THE PRICE COLUMN: "RM 1,250,000" → 1250000.0
df['Price'] = df['Price'].replace('[^0-9.]', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# 2. CLEAN THE SIZE COLUMN: "Built-up : 1,335 sq. ft." → 1335.0
df['Size'] = df['Size'].replace('[^0-9.]', '', regex=True)
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')

# 3. CLEAN THE ROOMS COLUMN: "3+1" → 3
df['Rooms'] = df['Rooms'].astype(str).str.extract(r'(\d+)').astype(float)

# 4. DROP ROWS WITH MISSING TARGET (Price)
df = df.dropna(subset=['Price'])
print(f"After dropping missing prices: {df.shape}")

# 5. HANDLE MISSING VALUES IN FEATURES
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())
df['Car Parks'] = df['Car Parks'].fillna(df['Car Parks'].median())
df['Size'] = df['Size'].fillna(df['Size'].median())
df['Rooms'] = df['Rooms'].fillna(df['Rooms'].median())
df['Furnishing'] = df['Furnishing'].fillna('Unknown')
df['Property Type'] = df['Property Type'].fillna('Unknown')

# 6. REMOVE OUTLIERS
df = df[df['Price'] > 0]
df = df[df['Size'] > 0]
print(f"After removing outliers: {df.shape}")

# 7. FEATURE ENGINEERING
df['Price_per_sqft'] = df['Price'] / df['Size']

# 8. ENCODE CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
categorical_cols = ['Location', 'Property Type', 'Furnishing']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"{col}: {len(le.classes_)} unique values")

# 9. PREPARE FINAL DATASET
feature_cols = ['Location_encoded', 'Rooms', 'Bathrooms', 'Car Parks',
                'Property Type_encoded', 'Size', 'Furnishing_encoded']

X = df[feature_cols]
y = df['Price']

print(f"\nFinal dataset:")
print(f"  Features (X): {X.shape}")
print(f"  Target (y): {y.shape}")
print(f"\nPrice statistics:")
print(y.describe())

# 10. SPLIT INTO TRAIN/TEST SETS (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Save the processed data for the next step
df.to_csv('data/kl_property_cleaned.csv', index=False)
print("\nSaved cleaned data to data/kl_property_cleaned.csv")
