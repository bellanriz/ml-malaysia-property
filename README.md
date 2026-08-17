# Malaysia Property Price Prediction

A machine learning regression project to predict property prices in Malaysia (Kuala Lumpur) using scikit-learn and python.

## Dataset

The dataset is sourced from Kaggle and contains property listings from Kuala Lumpur, Malaysia.

**Features:**
- Location: Neighborhood where the property is situated
- Rooms: Number of bedrooms
- Bathrooms: Number of bathrooms
- Car Parks: Number of parking spots
- Property Type: Type of property (Condominium, Apartment, etc.)
- Size (sq.ft): Total size of the property
- Furnishing: Furnishing status (Fully Furnished, Partly Furnished, Not Furnished)

**Target Variable:**
- Price (RM): Listed sales price in Malaysian Ringgit

## Project Structure

```
ml-malaysia-property/
├── data/
│   └── kl_property.csv          # Place your Kaggle dataset here
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   ├── train_model.py           # Model training & evaluation
│   └── predict.py               # Make predictions on new data
├── models/                      # Saved trained models
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Download the dataset from Kaggle and place it in `data/kl_property.csv`
2. Run preprocessing and training:

```bash
python src/train_model.py
```

3. Make predictions:

```bash
python src/predict.py
```

## Models Used

- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
