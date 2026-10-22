import streamlit as st
import numpy as np
import joblib

# Load saved model, scaler, and label encoder
model = joblib.load('best_bean_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Dry Bean Classifier App", page_icon="🌱", layout="centered")

st.title("🌱 Dry Bean Type Classification App")
st.write("Enter the physical characteristics of the bean below to predict its exact type using Machine Learning.")

st.sidebar.header("Input Features")

# Input fields for the 16 features
area = st.sidebar.number_input("Area (A)", min_value=0.0, value=28395.0)
perimeter = st.sidebar.number_input("Perimeter (P)", min_value=0.0, value=610.29)
major_axis = st.sidebar.number_input("Major Axis Length", min_value=0.0, value=208.17)
minor_axis = st.sidebar.number_input("Minor Axis Length", min_value=0.0, value=173.88)
aspect_ratio = st.sidebar.number_input("Aspect Ratio", min_value=0.0, value=1.19)
eccentricity = st.sidebar.number_input("Eccentricity", min_value=0.0, value=0.55)
convex_area = st.sidebar.number_input("Convex Area", min_value=0.0, value=28715.0)
equiv_diameter = st.sidebar.number_input("Equivalent Diameter", min_value=0.0, value=190.14)
extent = st.sidebar.number_input("Extent", min_value=0.0, value=0.76)
solidity = st.sidebar.number_input("Solidity", min_value=0.0, value=0.98)
roundness = st.sidebar.number_input("Roundness", min_value=0.0, value=0.95)
compactness = st.sidebar.number_input("Compactness", min_value=0.0, value=0.91)
sf1 = st.sidebar.number_input("ShapeFactor1", min_value=0.0, value=0.007)
sf2 = st.sidebar.number_input("ShapeFactor2", min_value=0.0, value=0.003)
sf3 = st.sidebar.number_input("ShapeFactor3", min_value=0.0, value=0.83)
sf4 = st.sidebar.number_input("ShapeFactor4", min_value=0.0, value=0.99)

if st.button("Classify Bean"):
    # Create input array
    input_data = np.array([[area, perimeter, major_axis, minor_axis, aspect_ratio, 
                            eccentricity, convex_area, equiv_diameter, extent, 
                            solidity, roundness, compactness, sf1, sf2, sf3, sf4]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    predicted_class = le.inverse_transform(prediction)[0]
    
    st.success(f"### Predicted Bean Type: **{predicted_class}**")