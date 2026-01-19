import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    """Load your exact trained model from notebook"""
    return joblib.load('house_price_model.pkl')

model = load_model()

# Page config
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 **House Price Predictor**")
st.markdown("*R² = 0.644 | GradientBoosting + GridSearchCV | Production Ready*")

# Your EXACT prediction function from notebook
@st.cache_data
def predict_price(area, bedrooms, bathrooms, stories, mainroad=1, guestroom=0, 
                 basement=0, hotwaterheating=0, airconditioning=0, parking=2, 
                 prefarea=0, furnishingstatus=1):
    """Predict house price using trained model - EXACT from your notebook"""
    input_df = pd.DataFrame({
        'area': [area], 'bedrooms': [bedrooms], 'bathrooms': [bathrooms],
        'stories': [stories], 'mainroad': [mainroad], 'guestroom': [guestroom],
        'basement': [basement], 'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning], 'parking': [parking],
        'prefarea': [prefarea], 'furnishingstatus': [furnishingstatus]
    })
    return model.predict(input_df)[0]

# Input sidebar
st.sidebar.header("🏠 **House Features**")
area = st.sidebar.slider("📏 Area (sqft)", 1000, 10000, 5000)
bedrooms = st.sidebar.slider("🛏️ Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("🚿 Bathrooms", 1, 4, 2)
stories = st.sidebar.slider("🏢 Stories", 1, 4, 2)
parking = st.sidebar.slider("🚗 Parking", 0, 3, 2)
furnishingstatus = st.sidebar.selectbox("🛋️ Furnishing", 
                                       [2, 1, 0], 
                                       format_func=lambda x: 
                                       ['furnished', 'semi-furnished', 'unfurnished'][x])

st.sidebar.header("📍 Location Features")
mainroad = st.sidebar.selectbox("🛣️ Main Road?", [1, 0], 
                               format_func=lambda x: ['Yes', 'No'][x])
prefarea = st.sidebar.selectbox("⭐ Preferred Area?", [1, 0], 
                               format_func=lambda x: ['Yes', 'No'][x])
guestroom = st.sidebar.checkbox("👥 Guest Room", value=False)
basement = st.sidebar.checkbox("🕳️ Basement", value=False)
hotwaterheating = st.sidebar.checkbox("🔥 Hot Water Heating", value=False)
airconditioning = st.sidebar.checkbox("❄️ Air Conditioning", value=False)

# Main prediction
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔮 **Predict Price**", type="primary", use_container_width=True):
        price = predict_price(area, bedrooms, bathrooms, stories, 
                             mainroad, guestroom, basement, 
                             hotwaterheating, airconditioning, 
                             parking, prefarea, furnishingstatus)
        
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 1rem; color: white;'>
            <h2>₹{price:,.0f}</h2>
            <p><strong>Your House Value</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        st.success(f"✅ **Prediction Complete!** Expected error: ±₹1.34M (RMSE)")

# Metrics sidebar
with st.sidebar.expander("📊 **Model Performance**"):
    st.metric("Test R²", "0.644")
    st.metric("CV R² (mean)", "0.539 ± 0.042")
    st.metric("Test RMSE", "₹1.34M")
    st.info("**Best Params**: max_depth=4, n_estimators=200")

# Footer
st.markdown("---")
st.markdown("*Built with your exact notebook model | Deployed via Streamlit*")
