import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model()
    return joblib.load('house_price_model.pkl')

model = load_model()

st.title( House Price Predictor)
st.markdown(R² = 0.633  GradientBoosting + Cross-Validation)

col1, col2 = st.columns(2)
with col1
    area = st.slider( Area (sqft), 1000, 10000, 5000)
    bedrooms = st.slider( Bedrooms, 1, 6, 3)
with col2
    bathrooms = st.slider( Bathrooms, 1, 4, 2)
    parking = st.slider( Parking, 0, 3, 2)

mainroad = st.selectbox( Main Road, ['yes', 'no'])
prefarea = st.selectbox( Preferred Area, ['yes', 'no'])
furnishing = st.selectbox( Furnishing, 
                         ['furnished', 'semi-furnished', 'unfurnished'])

if st.button(🔮 Predict Price, type=primary)
    input_data = pd.DataFrame({
        'area' [area], 'bedrooms' [bedrooms], 'bathrooms' [bathrooms],
        'stories' [2], 'mainroad' [mainroad], 'guestroom' ['no'],
        'basement' ['no'], 'hotwaterheating' ['no'], 'airconditioning' ['no'],
        'parking' [parking], 'prefarea' [prefarea], 'furnishingstatus' [furnishing]
    })
    price = model.predict(input_data)[0]
    st.success(fPredicted Price ₹{price,.0f})
    st.balloons()
