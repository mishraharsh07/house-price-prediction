import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)

# Configuration for Realistic Indian Data
cities = {
    'Mumbai': {'avg_price_sqft': 22000, 'pci': 4.13, 'tier': 1, 'volatility': 0.3},
    'Delhi NCR': {'avg_price_sqft': 10500, 'pci': 4.75, 'tier': 1, 'volatility': 0.25},
    'Bangalore': {'avg_price_sqft': 11500, 'pci': 9.80, 'tier': 1, 'volatility': 0.2},
    'Hyderabad': {'avg_price_sqft': 9500, 'pci': 5.54, 'tier': 2, 'volatility': 0.2},
    'Pune': {'avg_price_sqft': 9500, 'pci': 2.78, 'tier': 2, 'volatility': 0.15},
    'Chennai': {'avg_price_sqft': 8500, 'pci': 3.50, 'tier': 2, 'volatility': 0.15},
    'Kolkata': {'avg_price_sqft': 7500, 'pci': 2.50, 'tier': 2, 'volatility': 0.1},
    'Ahmedabad': {'avg_price_sqft': 6500, 'pci': 2.80, 'tier': 2, 'volatility': 0.1},
    'Jaipur': {'avg_price_sqft': 5500, 'pci': 1.80, 'tier': 3, 'volatility': 0.1},
    'Lucknow': {'avg_price_sqft': 5000, 'pci': 1.60, 'tier': 3, 'volatility': 0.1}
}

num_samples = 15000
data = []

for _ in range(num_samples):
    city = random.choice(list(cities.keys()))
    c_info = cities[city]
    
    # Square footage based on Indian house types (BHKs)
    bhk = random.choice([1, 2, 3, 4, 5])
    if bhk == 1: sqft = random.randint(400, 700)
    elif bhk == 2: sqft = random.randint(800, 1200)
    elif bhk == 3: sqft = random.randint(1300, 2000)
    elif bhk == 4: sqft = random.randint(2200, 4000)
    else: sqft = random.randint(4500, 10000)
    
    # Amenities (Common in Indian gated societies)
    has_pool = 1 if (random.random() < 0.3 or c_info['tier'] == 1) else 0
    has_gym = 1 if (random.random() < 0.5 or has_pool) else 0
    is_gated = 1 if (random.random() < 0.6) else 0
    
    # Distance from center (Km)
    dist_center = random.uniform(0, 30)
    
    # Floor number (High rise is premium in Mumbai/Bangalore)
    floor = random.randint(0, 40 if c_info['tier'] == 1 else 15)
    
    # Base Price calculation
    # Price = (Sqft * BaseRate) * (IncomeFactor) * (AmenityFactor) * (LocationFactor)
    base_rate = c_info['avg_price_sqft']
    
    # Add noise
    base_rate *= (1 + np.random.normal(0, c_info['volatility']))
    
    # Location penalty
    loc_factor = 1.0 / (1 + (dist_center * 0.03))
    
    # Floor premium
    floor_factor = 1 + (floor * 0.005)
    
    # Amenity premium
    amenity_factor = 1 + (has_pool * 0.15) + (has_gym * 0.1) + (is_gated * 0.05)
    
    # Size premium (Luxury factor for very large houses)
    size_factor = 1.0 + (max(0, sqft - 3000) * 0.0001)
    
    price_inr = sqft * base_rate * loc_factor * floor_factor * amenity_factor * size_factor
    
    # Ensure realistic minimums (e.g., 20 Lakhs)
    price_inr = max(2000000, price_inr)
    
    data.append({
        'City': city,
        'BHK': bhk,
        'Sqft': sqft,
        'Bathrooms': bhk if random.random() > 0.3 else bhk - 1,
        'Floor': floor,
        'Total_Floors': floor + random.randint(0, 10),
        'Distance_to_Center': round(dist_center, 2),
        'Has_Swimming_Pool': has_pool,
        'Has_Gym': has_gym,
        'Is_Gated_Community': is_gated,
        'Per_Capita_Income_Lakh': c_info['pci'],
        'Price_INR': round(price_inr, 0)
    })

df = pd.DataFrame(data)

# Save the dataset
save_path = r'C:\Users\ASUS\house-price-prediction\indian_house_prices.csv'
df.to_csv(save_path, index=False)

print(f"Generated {num_samples} realistic Indian house price records.")
print(f"Dataset saved to: {save_path}")
print(df['Price_INR'].describe())
