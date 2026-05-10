# House Price Prediction Model

This project implements a high-accuracy Machine Learning model to predict house prices based on various features like location, square footage, number of bedrooms, and condition.

## Features
- **Algorithm:** HistGradientBoostingRegressor (Gradient Boosting)
- **R2 Score:** 0.7419
- **Mean Absolute Error (MAE):** ~$87,000
- **Feature Engineering:** 
  - Log transformation of the target variable for better handling of skewed distributions.
  - Outlier removal (top 1% of prices) to improve model robustness.
  - Extraction of house age and renovation status.
  - Aggregation of total square footage.
  - Label encoding for geographical data (city, zip code).

## Project Structure
- `data.csv`: The raw dataset.
- `train_model.py`: Script to preprocess data, engineer features, and train the model.
- `predict.py`: Script to make predictions using the trained model.
- `house_price_model.pkl`: The trained model file.
- `le_city.pkl` & `le_statezip.pkl`: Label encoders for categorical data.

## How to Run
1. Run `python train_model.py` to retrain the model.
2. Run `python predict.py` to see a sample prediction.

## Dataset
The dataset includes historical house sales with features like:
- Bedrooms/Bathrooms
- Sqft Living/Lot
- Year Built/Renovated
- City/Statezip
- Waterfront/View status
