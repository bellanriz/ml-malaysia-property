"""
Step 5: Visualization & EDA Charts
- Price distribution
- Feature correlations
- Model comparison
- Feature importance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('data/kl_property_cleaned.csv')
print(f"Loaded: {df.shape}")

# CHART 1: Price Distribution (Before vs After Log Transform)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Original price
axes[0].hist(df['Price'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Price Distribution (Original)')
axes[0].set_xlabel('Price (RM)')
axes[0].set_ylabel('Count')

# Right: Log-transformed price
axes[1].hist(df['Price_log'], bins=50, color='coral', edgecolor='black')
axes[1].set_title('Price Distribution (Log Transformed)')
axes[1].set_xlabel('Log(Price)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('data/chart1_price_distribution.png', dpi=100)
plt.close()
print("Saved: chart1_price_distribution.png")

# CHART 2: Correlation Heatmap
numeric_cols = ['Price', 'Rooms', 'Bathrooms', 'Car Parks', 'Size']
correlation = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('data/chart2_correlation.png', dpi=100)
plt.close()
print("Saved: chart2_correlation.png")

# CHART 3: Price vs Size (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(df['Size'], df['Price'], alpha=0.3, s=10, color='steelblue')
plt.xlabel('Size (sq. ft.)')
plt.ylabel('Price (RM)')
plt.title('Price vs Size')
plt.tight_layout()
plt.savefig('data/chart3_price_vs_size.png', dpi=100)
plt.close()
print("Saved: chart3_price_vs_size.png")

# CHART 4: Average Price by Furnishing
furnishing_avg = df.groupby('Furnishing')['Price'].mean().sort_values()

plt.figure(figsize=(8, 5))
furnishing_avg.plot(kind='barh', color='teal', edgecolor='black')
plt.xlabel('Average Price (RM)')
plt.ylabel('Furnishing')
plt.title('Average Price by Furnishing Status')
plt.tight_layout()
plt.savefig('data/chart4_furnishing.png', dpi=100)
plt.close()
print("Saved: chart4_furnishing.png")

# CHART 5: Top 10 Most Expensive Locations
location_avg = df.groupby('Location')['Price'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
location_avg.plot(kind='barh', color='goldenrod', edgecolor='black')
plt.xlabel('Average Price (RM)')
plt.ylabel('Location')
plt.title('Top 10 Most Expensive Locations')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('data/chart5_top_locations.png', dpi=100)
plt.close()
print("Saved: chart5_top_locations.png")

print("\nAll charts saved to data/ folder!")
print("Open them in Finder to view.")
