import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Load the dataset
data_path = r'C:\Users\ASUS\house-price-prediction\data.csv'
df = pd.read_csv(data_path)

# 1. Advanced Cleaning
df = df[df['price'] > 0] 
# Tighter outlier removal for "High Accuracy"
q_low = df["price"].quantile(0.01)
q_hi = df["price"].quantile(0.98)
df = df[(df["price"] > q_low) & (df["price"] < q_hi)]

df.drop(['country', 'street'], axis=1, errors='ignore', inplace=True)

# 2. Enhanced Feature Engineering
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df.drop('date', axis=1, inplace=True)

df['house_age'] = df['year'] - df['yr_built']
df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
df['total_sqft'] = df['sqft_living'] + df['sqft_above'] + df['sqft_basement']

# Feature Interactions
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
df['sqft_per_room'] = df['total_sqft'] / (df['bedrooms'] + df['bathrooms'] + 1)

# 3. Encoding
le_city = LabelEncoder()
df['city'] = le_city.fit_transform(df['city'])

le_statezip = LabelEncoder()
df['statezip'] = le_statezip.fit_transform(df['statezip'])

# 4. Split and Scale
X = df.drop('price', axis=1)
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 5. Stacking Regressor for Maximum Accuracy
estimators = [
    ('hgb', HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.02, max_depth=15, l2_regularization=0.5, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42))
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV()
)

print("Training high-accuracy stacking model (this may take a minute)...")
stacking_model.fit(X_train, y_train)

# 6. Evaluation
y_pred_log = stacking_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

r2 = r2_score(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

print(f"Final High-Accuracy Model Performance:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")

# 7. Save Model, Encoders, and Feature Names
save_dir = r'C:\Users\ASUS\house-price-prediction'
joblib.dump(stacking_model, os.path.join(save_dir, 'house_price_model.pkl'))
joblib.dump(le_city, os.path.join(save_dir, 'le_city.pkl'))
joblib.dump(le_statezip, os.path.join(save_dir, 'le_statezip.pkl'))
joblib.dump(list(X.columns), os.path.join(save_dir, 'feature_names.pkl'))

print(f"Model and feature names saved successfully with R2: {r2:.4f}")
