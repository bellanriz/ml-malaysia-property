"""
Step 2: Data Preprocessing (Improved)
- Clean Price, Size, Rooms columns
- Remove extreme outliers
- Log-transform Price (handles skewed distribution)
- Use better encoding for Location
"""
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/kl_property.csv')
print(f"Original shape: {df.shape}")

# 1. CLEAN THE PRICE COLUMN: "RM 1,250,000" → 1250000.0
df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False)
df['Price'] = df['Price'].str.extract(r'(\d+\.?\d*)').astype(float)

# 2. CLEAN THE SIZE COLUMN: "Built-up : 1,335 sq. ft." → 1335.0
#    First remove commas, then extract the first number (digits and optional decimal)
df['Size'] = df['Size'].astype(str).str.replace(',', '', regex=False)
df['Size'] = df['Size'].str.extract(r'(\d+\.?\d*)').astype(float)

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

# 6. REMOVE OUTLIERS (this is the key fix!)
#    Keep only properties between RM 100k and RM 10 million
#    Keep only sizes between 200 and 10,000 sq.ft
df = df[df['Price'] > 0]
df = df[df['Size'] > 0]
df = df[(df['Price'] >= 100000) & (df['Price'] <= 10000000)]
df = df[(df['Size'] >= 200) & (df['Size'] <= 10000)]
print(f"After removing outliers: {df.shape}")

# 7. LOG-TRANSFORM PRICE (key improvement!)
#    Price is heavily skewed — log makes it more normally distributed
#    This helps linear models enormously
df['Price_log'] = np.log1p(df['Price'])

# 8. FEATURE ENGINEERING
df['Price_per_sqft'] = df['Price'] / df['Size']

# 9. ENCODE CATEGORICAL VARIABLES
#    Use Label Encoding for tree-based models
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
categorical_cols = ['Location', 'Property Type', 'Furnishing']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"{col}: {len(le.classes_)} unique values")

# 10. PREPARE FINAL DATASET
feature_cols = ['Location_encoded', 'Rooms', 'Bathrooms', 'Car Parks',
                'Property Type_encoded', 'Size', 'Furnishing_encoded']

X = df[feature_cols]
y = df['Price_log']  # Using log-transformed price!

print(f"\nFinal dataset:")
print(f"  Features (X): {X.shape}")
print(f"  Target (y): {y.shape}")
print(f"\nPrice statistics (original):")
print(df['Price'].describe())

# Save
df.to_csv('data/kl_property_cleaned.csv', index=False)
print("\nSaved cleaned data to data/kl_property_cleaned.csv")
