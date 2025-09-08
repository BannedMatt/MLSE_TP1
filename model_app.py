import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression

model = joblib.load("regression.joblib")

size = st.number_input("how many big")
number_of_bedrooms = st.number_input("number of bedrooms")
garden = st.number_input("garden or not")

st.write(model.predict([[size, number_of_bedrooms, garden]])[0])