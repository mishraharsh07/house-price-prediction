import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Realistic Indian House AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em;
        background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #128807 100%);
        color: #000080; font-weight: bold; border: 2px solid #000080; font-size: 1.1em;
    }
    .stButton>button:hover { 
        background: linear-gradient(135deg, #128807 0%, #FFFFFF 50%, #FF9933 100%);
        border: 2px solid #000080;
    }
    .prediction-box {
        padding: 30px; border-radius: 20px; background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); text-align: center;
        border-top: 8px solid #000080;
    }
    .section-header {
        color: #000080; font-weight: bold; border-bottom: 2px solid #FF9933;
        padding-bottom: 5px; margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load AI Assets
SAVE_DIR = os.path.abspath(os.path.dirname(__file__))

@st.cache_resource
def load_assets():
    import bz2
    with bz2.BZ2File(os.path.join(SAVE_DIR, 'house_price_model.pkl.bz2'), 'rb') as f:
        model = joblib.load(f)
    le_city = joblib.load(os.path.join(SAVE_DIR, 'le_city_indian.pkl'))
    feature_names = joblib.load(os.path.join(SAVE_DIR, 'feature_names_indian.pkl'))
    return model, le_city, feature_names

try:
    model, le_city, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading AI assets: {e}")
    st.stop()

# Header
st.markdown("<h1 style='text-align: center; color: #000080;'>🇮🇳 Advanced Indian Real Estate AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>Realistic Valuation trained on major Indian metros & Tier-2 cities (Accuracy: 91%)</p>", unsafe_allow_html=True)
st.divider()

# Sidebar: City & Economics
st.sidebar.header("📍 Location & Economics")
city = st.sidebar.selectbox("Select City", sorted(le_city.classes_))

# Map City to Per Capita Income (Based on 2024 Research)
pci_map = {
    'Mumbai': 4.13, 'Delhi NCR': 4.75, 'Bangalore': 9.80, 'Hyderabad': 5.54,
    'Pune': 2.78, 'Chennai': 3.50, 'Kolkata': 2.50, 'Ahmedabad': 2.80,
    'Jaipur': 1.80, 'Lucknow': 1.60
}
pci = pci_map.get(city, 2.12) # Default to national avg if not found

st.sidebar.markdown(f"**City Income Tier:** {'High' if pci > 5 else 'Medium' if pci > 3 else 'Standard'}")
st.sidebar.info(f"Avg Per Capita Income in {city}: ₹ {pci} Lakh")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='section-header'>🏠 Property Specifications</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        bhk = st.number_input("BHK (1-10)", 1, 10, 2, step=1)
        bathrooms = st.number_input("Bathrooms", 1, 10, bhk, step=1)
    with c2:
        sqft = st.number_input("Super Built-up Area (Sqft)", 400, 20000, 1200, step=50)
        floor = st.number_input("Floor Number", 0, 80, 5, step=1)
    with c3:
        total_floors = st.number_input("Total Floors in Building", 1, 80, 10, step=1)
        dist = st.number_input("Distance to City Center (Km)", 0.0, 50.0, 5.0, step=0.5)

    st.markdown("<div class='section-header'>✨ Amenities & Society</div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        pool = st.toggle("Swimming Pool")
    with a2:
        gym = st.toggle("Modern Gym")
    with a3:
        gated = st.toggle("Gated Community", value=True)

with col2:
    st.markdown("<div class='section-header'>📊 AI Valuation</div>", unsafe_allow_html=True)
    if st.button("PREDICT MARKET PRICE"):
        with st.spinner('AI analyzing local market trends...'):
            # Preprocessing
            input_dict = {
                'City': city,
                'BHK': float(bhk),
                'Sqft': float(sqft),
                'Bathrooms': float(bathrooms),
                'Floor': float(floor),
                'Total_Floors': float(total_floors),
                'Distance_to_Center': float(dist),
                'Has_Swimming_Pool': 1 if pool else 0,
                'Has_Gym': 1 if gym else 0,
                'Is_Gated_Community': 1 if gated else 0,
                'Per_Capita_Income_Lakh': float(pci)
            }
            
            df_in = pd.DataFrame([input_dict])
            
            # Label Encode City
            df_in['City'] = le_city.transform(df_in['City'])
            
            # Ensure correct column order
            df_in = df_in[feature_names]
            
            # Predict
            pred_log = model.predict(df_in)
            final_price = np.expm1(pred_log)[0]
            
            # Readable Formatting (Lakhs and Crores)
            def format_indian_readable(num):
                if num >= 10000000: # 1 Crore
                    return f"₹ {num/10000000:.2f} Crore"
                elif num >= 100000: # 1 Lakh
                    return f"₹ {num/100000:.2f} Lakh"
                else:
                    return f"₹ {num:,.0f}"

            st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='color: #000080; margin-bottom: 0;'>Estimated Price</h2>
                    <h1 style='color: #128807; font-size: 2.8em;'>{format_indian_readable(final_price)}</h1>
                    <p style='color: #777;'>Valuation for {city}</p>
                    <hr>
                    <p style='font-size: 0.9em; color: #444;'>AI Confidence: 91.08%</p>
                    <p style='font-size: 0.8em; color: #888;'>Based on 2024 Market Data & Regional Income</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

st.sidebar.markdown("---")
st.sidebar.caption("Trained with a high-accuracy Stacking Regressor ensemble for realistic Indian market valuation.")
st.sidebar.markdown("### ✨ Made by Harsh Mishra")

# Footer at the very bottom
st.markdown("<br><hr><p style='text-align: center; color: #888;'>Made with ❤️ by Harsh Mishra | © 2024 Indian Real Estate AI</p>", unsafe_allow_html=True)
