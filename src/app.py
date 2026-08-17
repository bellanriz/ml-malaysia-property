import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============================================================
# PAGE CONFIG (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="KL Property Predictor",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
# CUSTOM CSS (makes it look nicer)
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .prediction-price {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/kl_property_cleaned.csv')

model = load_model()
df = load_data()

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">🏠 KL Property Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict property prices in Kuala Lumpur using Machine Learning</p>', unsafe_allow_html=True)

st.divider()

# ============================================================
# SIDEBAR - Project Info
# ============================================================
with st.sidebar:
    st.header("📊 About This Project")
    st.write("""
    This app predicts property prices in Kuala Lumpur
    using a machine learning model trained on **{:,}** 
    real property listings.
    """.format(len(df)))

    st.divider()

    st.subheader("🤖 Model Info")
    st.write(f"**Algorithm:** {type(model).__name__}")
    st.write(f"**Features:** 7")
    st.write(f"**Training Data:** {len(df):,} properties")


    st.divider()

    st.subheader("📈 Price Range in Dataset")
    st.write(f"**Min:** RM {df['Price'].min():,.0f}")
    st.write(f"**Max:** RM {df['Price'].max():,.0f}")
    st.write(f"**Average:** RM {df['Price'].mean():,.0f}")

# ============================================================
# MAIN CONTENT - Two Columns
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Property Details")

    location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
    property_type = st.selectbox("🏢 Property Type", sorted(df['Property Type'].unique()))
    furnishing = st.selectbox("🛋️ Furnishing", sorted(df['Furnishing'].unique()))
    size = st.number_input("📐 Size (sq.ft)", min_value=200, max_value=10000, value=1000, step=50)

with col2:
    st.subheader("🔢 Specifications")

    rooms = st.slider("🛏️ Bedrooms", 1, 10, 3)
    bathrooms = st.slider("🚿 Bathrooms", 1, 10, 2)
    car_parks = st.slider("🚗 Car Parks", 0, 5, 2)

    # Show a summary of what the user selected
    st.markdown("---")
    st.markdown("**Your Selection:**")
    st.write(f"📍 {location}")
    st.write(f"📐 {size:,} sq.ft | 🛏️ {rooms} bed | 🚿 {bathrooms} bath | 🚗 {car_parks} parks")

# ============================================================
# PREDICTION
# ============================================================
st.divider()

# Encode inputs
location_encoded = df[df['Location'] == location]['Location_encoded'].iloc[0]
type_encoded = df[df['Property Type'] == property_type]['Property Type_encoded'].iloc[0]
furnishing_encoded = df[df['Furnishing'] == furnishing]['Furnishing_encoded'].iloc[0]

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        features = pd.DataFrame({
            'Location_encoded': [location_encoded],
            'Rooms': [rooms],
            'Bathrooms': [bathrooms],
            'Car Parks': [car_parks],
            'Property Type_encoded': [type_encoded],
            'Size': [size],
            'Furnishing_encoded': [furnishing_encoded],
        })

        prediction_log = model.predict(features)
        price = np.expm1(prediction_log[0])

        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <p style="margin:0; font-size:1rem;">Estimated Property Price</p>
            <p class="prediction-price">RM {price:,.0f}</p>
            <p style="margin:0; font-size:0.9rem;">Based on {type(model).__name__} model</p>
        </div>
        """, unsafe_allow_html=True)

        # Show price per sqft
        price_per_sqft = price / size
        st.markdown("")

        m1, m2, m3 = st.columns(3)
        m1.metric("Price per sq.ft", f"RM {price_per_sqft:,.0f}")
        m2.metric("Monthly (30yr loan)", f"RM {price * 0.004:,.0f}")
        m3.metric("Down Payment (10%)", f"RM {price * 0.1:,.0f}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<p style="text-align:center; color:#999; font-size:0.8rem;">
Built with Streamlit & scikit-learn | Dataset: Kaggle (Property Listings in KL)
</p>
""", unsafe_allow_html=True)
