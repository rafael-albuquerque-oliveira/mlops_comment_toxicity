import streamlit as st
import requests

API_URL = "http://localhost:3000/classify"

st.title("Toxicity Classifier")
text_input = st.text_area("Enter text in Portuguese:")

if st.button("Classify"):
    response = requests.post(API_URL, json=text_input)
    if response.status_code == 200:
        st.write("Prediction:", response.json()["prediction"])
    else:
        st.write("Error:", response.text)
