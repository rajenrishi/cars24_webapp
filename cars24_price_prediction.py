import pickle

import yfinance as yf
import streamlit as st
import pandas as pd

df_cars = pd.read_csv("cars24-car-price.csv")

st.dataframe(df_cars.head())

# Encoding categorical features
encode_dict = {
    "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3},
    "seller_type": {"Dealer": 1, "Individual": 2},
    "transmission_type": {"Manual": 1, "Automatic": 2}
}

col1, col2 = st.columns(2)

fuel_type = col1.selectbox("Select Fuel Type: ", ["Diesel", "Petrol", "CNG"])
engine = col1. slider("Set the Engine Power:", 500, 5000, step=500)
transmission_type = col2.selectbox("Select Transmission Type:", ["Manual", "Automatic"])
number_of_seats = col2.selectbox("Number of seats:", [4, 5, 7, 9, 11])

fuel_type = encode_dict["fuel_type"][fuel_type]
transmission_type = encode_dict["transmission_type"][transmission_type]
   

def model_pred(fuel_type_val, engine_val, transmission_type_val, number_of_seats_val):
    with open("car_pred", "rb") as file:
        reg_model = pickle.load(file)
        features = [[2012, 2, 12000, fuel_type_val, transmission_type_val, 19.7, engine_val, 46.3, number_of_seats_val]]
        return reg_model.predict(features)

if st.button("Predict Price"):
    predicted_price = model_pred(fuel_type, engine, transmission_type, number_of_seats)
    st.text(predicted_price)
