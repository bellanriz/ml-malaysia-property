"""
Step 2: Exploratory Data Analysis (EDA)

Run this after load_data.py to understand the dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/kl_property.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nStatistical Summary:\n{df.describe()}")

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n" + "=" * 50)
print("CATEGORICAL COLUMNS - UNIQUE VALUES")
print("=" * 50)
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(10))

# --- Visualizations ---

# 1. Price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Distribution of Property Prices (RM)')
plt.xlabel('Price (RM)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("data/price_distribution.png")
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("data/correlation_heatmap.png")
plt.show()

# 3. Price by property type (if column exists)
if 'Property Type' in df.columns:
    plt.figure(figsize=(12, 5))
    top_types = df['Property Type'].value_counts().head(10).index
    sns.boxplot(data=df[df['Property Type'].isin(top_types)],
                x='Property Type', y='Price')
    plt.xticks(rotation=45, ha='right')
    plt.title('Price by Property Type')
    plt.tight_layout()
    plt.savefig("data/price_by_type.png")
    plt.show()

print("\nEDA complete! Check the plots above.")
