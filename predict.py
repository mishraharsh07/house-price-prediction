import pandas as pd
import numpy as np
import joblib
import os

def predict_house_price(data_dict):
    save_dir = r'C:\Users\ASUS\house-price-prediction'
    model = joblib.load(os.path.join(save_dir, 'house_price_model.pkl'))
    le_city = joblib.load(os.path.join(save_dir, 'le_city.pkl'))
    le_statezip = joblib.load(os.path.join(save_dir, 'le_statezip.pkl'))
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Feature Engineering (must match training)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.drop('date', axis=1, inplace=True)
    
    df['house_age'] = df['year'] - df['yr_built']
    df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    df['years_since_renovation'] = df.apply(lambda x: x['year'] - x['yr_renovated'] if x['yr_renovated'] > 0 else x['house_age'], axis=1)
    df['total_sqft'] = df['sqft_living'] + df['sqft_above'] + df['sqft_basement']
    
    # Encode
    df['city'] = le_city.transform(df['city'])
    df['statezip'] = le_statezip.transform(df['statezip'])
    
    # Predict
    log_price = model.predict(df)
    price = np.expm1(log_price)[0]
    
    return price

if __name__ == "__main__":
    # Example usage
    sample_house = {
        'date': '2014-05-02',
        'bedrooms': 3.0,
        'bathrooms': 2.0,
        'sqft_living': 1930,
        'sqft_lot': 11947,
        'floors': 1.0,
        'waterfront': 0,
        'view': 0,
        'condition': 4,
        'sqft_above': 1930,
        'sqft_basement': 0,
        'yr_built': 1966,
        'yr_renovated': 0,
        'city': 'Kent',
        'statezip': 'WA 98042'
    }
    
    predicted_price = predict_house_price(sample_house)
    print(f"Predicted Price for the sample house: ${predicted_price:,.2f}")
