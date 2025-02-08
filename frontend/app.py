import streamlit as st
import requests

# Define the API URL (replace with your actual Cloud Run URL)
API_URL = "https://toxic-comments-api-678895434688.europe-west4.run.app/predict"

st.title("🚀 Toxic Comment Detector")
st.write("Enter a comment below to analyze its toxicity.")

user_input = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if user_input.strip():  # Ensure input is not empty
        try:
            response = requests.post(API_URL, json={"text": user_input}, timeout=60)
            if response.status_code == 200:
                prediction = response.json()
                st.write("### Prediction Results:")
                st.json(prediction)  # Display JSON results
            else:
                st.error(f"⚠️ Error: API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ API Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a comment before analyzing.")

if __name__ == "__main__":
    # Ensure Streamlit is bound to the correct port
    import os
    os.system("streamlit run app.py --server.port=8501 --server.address=0.0.0.0")
