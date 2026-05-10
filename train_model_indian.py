import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Load the realistic Indian dataset
data_path = r'C:\Users\ASUS\house-price-prediction\indian_house_prices.csv'
df = pd.read_csv(data_path)

# 1. Encoding
le_city = LabelEncoder()
df['City'] = le_city.fit_transform(df['City'])

# 2. Split Data
X = df.drop('Price_INR', axis=1)
y = np.log1p(df['Price_INR']) # Log transform for better accuracy on wide range

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 3. High-Accuracy Stacking Model
estimators = [
    ('hgb', HistGradientBoostingRegressor(max_iter=2000, learning_rate=0.03, max_depth=15, l2_regularization=0.5, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42))
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV()
)

print("Training Advanced Indian House Price AI...")
stacking_model.fit(X_train, y_train)

# 4. Evaluation
y_pred_log = stacking_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

r2 = r2_score(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)

print(f"Model Accuracy (R2): {r2:.4f}")
print(f"Mean Absolute Error: ₹ {mae:,.0f}")

# 5. Save Model and Assets
save_dir = r'C:\Users\ASUS\house-price-prediction'
joblib.dump(stacking_model, os.path.join(save_dir, 'house_price_model.pkl'))
joblib.dump(le_city, os.path.join(save_dir, 'le_city_indian.pkl'))
joblib.dump(list(X.columns), os.path.join(save_dir, 'feature_names_indian.pkl'))

print("All AI assets for the Indian market saved successfully.")
