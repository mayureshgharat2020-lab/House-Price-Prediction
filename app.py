
import streamlit as st
import pandas as pd
import joblib

# Load Models
linear = joblib.load("linear_model.pkl")
poly_model = joblib.load("polynomial_model.pkl")
poly = joblib.load("poly_features.pkl")

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction System")

st.write("Predict house prices using Machine Learning")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Linear Regression", "Polynomial Regression")
)

st.header("Enter House Details")

area = st.number_input("Area (sq ft)", 500, 10000, 1500)

bedrooms = st.number_input("Bedrooms", 1, 10, 3)

bathrooms = st.number_input("Bathrooms", 1, 10, 2)

stories = st.number_input("Stories", 1, 5, 2)

parking = st.number_input("Parking", 0, 5, 1)

mainroad = st.selectbox("Main Road", ["yes", "no"])

guestroom = st.selectbox("Guest Room", ["yes", "no"])

basement = st.selectbox("Basement", ["yes", "no"])

hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])

airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])

prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# Encoding
yes_no = {
    "yes": 1,
    "no": 0
}

furnish = {
    "furnished": 0,
    "semi-furnished": 1,
    "unfurnished": 2
}

input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [yes_no[mainroad]],
    "guestroom": [yes_no[guestroom]],
    "basement": [yes_no[basement]],
    "hotwaterheating": [yes_no[hotwaterheating]],
    "airconditioning": [yes_no[airconditioning]],
    "parking": [parking],
    "prefarea": [yes_no[prefarea]],
    "furnishingstatus": [furnish[furnishingstatus]]
})

if st.button("Predict Price"):

    if model_choice == "Linear Regression":

        prediction = linear.predict(input_data)

    else:
        poly_input = poly.transform(input_data)

        prediction = poly_model.predict(poly_input)

    st.success(f"Predicted House Price : ₹ {prediction[0]:,.2f}")