import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained Decision Tree model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please run the training notebook first.")

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Machine Learning Car Price Predictor")
st.markdown("This AI application predicts the **Selling Price** of a used car based on its features.")

# Create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Present Showroom Price (in Million Tshs)", min_value=0.1, value=5.0)
    kms_driven = st.number_input("Total Kilometers Driven", min_value=0, value=20000)
    owner = st.selectbox("Number of Previous Owners", [0, 1, 3])
    age = st.number_input("Age of the Car (Years)", min_value=0, max_value=30, value=5)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Button to trigger prediction
if st.button("Calculate Predicted Selling Price"):
    # Preprocess inputs to match the model's training format (One-Hot Encoding)
    # The model expects: [Present_Price, Kms_Driven, Owner, Car_Age, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual]
    
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_petrol = 1 if fuel_type == "Petrol" else 0
    seller_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    input_data = np.array([[
        present_price, 
        kms_driven, 
        owner, 
        age, 
        fuel_diesel, 
        fuel_petrol, 
        seller_individual, 
        transmission_manual
    ]])

    prediction = model.predict(input_data)
    
    st.success(f"### The estimated Selling Price is: {prediction[0]:.2f} Million Tshs")
    st.info("Note: Prices are estimated based on historical trends in the dataset.")