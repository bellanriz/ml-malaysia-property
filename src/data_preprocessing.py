import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/kl_property.csv')
print(f"Original shape: {df.shape}")

df ['Price'] = df['Price'].replace('[^0-9]','', regex=True)
df ['Price'] = pd.to.numeric(df['Size'], errors='cource')

df = df.dropna(subset=['Price'])
print (f"After dropping missing prices: {df.shape}")

df ['Bathrooms']= df['Bathroom'].fillna(df['Bathrooms'].median())
df ['Car Parks']= df['Car Parks'].fillna(df['Car Parks'].median())
df ['Size']= df['Size'].fillna(df['Size'].median())
df ['Rooms']= df['Rooms'].fillna(df['Rooms'].median())
df ['Furnishing']= df['Furnishing'].fillna('Unknown')
df ['Property Type'] = df ['Property Type'].fillna('Unknown')

df = df[df['Price'] > 0]
df = df[df['Size'] > 0]
print (f"After removing outliners : {df.shape}")

df ['Price_per_sqft'] = df['Price'] / df['Size']

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
categorical_cols = ['Location', 'Property Type', 'Furnishing']


for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"{col} : {len(le.classes_)} unique values")

feature_cols =

x = df[feature_cols]
y = df['Price']

print(f"\nFinal dataset:")
print(f"   Feature (x): {x.shape}")
print(f"   Target (y): {y.shape}")
print(f"\n Price statistic:")
print(y.describe())

form sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Save the processed data for the next step
df.to_csv('data/kl_property_cleaned.csv', index=False)
print("\nSaved cleaned data to data/kl_property_cleaned.csv")

