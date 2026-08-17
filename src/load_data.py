"""
Step 1: Load the Malaysia Property dataset from Kaggle
"""
import kagglehub
import pandas as pd
import os

# Download the dataset (cached locally after first download)
path = kagglehub.dataset_download("dragonduck/property-listings-in-kuala-lumpur")
print(f"Dataset downloaded to: {path}")

# Find and load the CSV file
files = os.listdir(path)
print(f"Files in dataset: {files}")

csv_file = [f for f in files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Quick look at the data
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Save a copy to our data folder for easy access
os.makedirs("data", exist_ok=True)
df.to_csv("data/kl_property.csv", index=False)
print(f"\nSaved to data/kl_property.csv")
