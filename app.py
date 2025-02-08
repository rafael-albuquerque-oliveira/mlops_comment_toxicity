import streamlit as st
import requests

# Need to make it prettier
st.title("Toxic Comment Analyzer")

user_input = st.text_area("Enter a comment:")
if st.button("Analyze"):
    response = requests.post("https://toxic-comments-api-678895434688.us-central1.run.app/predict", json={"text": user_input})
    prediction = response.json()
    st.write(prediction)
