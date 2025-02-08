import streamlit as st
import requests

# Define the API URL (Replace with your actual Cloud Run URL)
API_URL = "https://toxic-comments-api-678895434688.us-central1.run.app/predict"

st.title("🚀 Toxic Comment Detector")
st.write("Enter a comment below to analyze its toxicity.")

# Input text area
user_input = st.text_area("Enter a comment:", "")

if st.button("Analyze"):
    if user_input.strip():  # Ensure input is not empty
        # Send request to API
        response = requests.post(API_URL, json={"text": user_input})

        if response.status_code == 200:
            prediction = response.json()
            st.write("### Prediction Results:")
            st.json(prediction)  # Display JSON results

        else:
            st.error("⚠️ Error: Unable to connect to API. Please try again later.")
    else:
        st.warning("⚠️ Please enter a comment before analyzing.")
