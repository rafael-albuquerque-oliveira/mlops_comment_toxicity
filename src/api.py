from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from src.translator import Translator
from src.data_loader import DataLoader  # Ensure correct import path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch
import joblib

def load_model():
    """Load the model only when needed to save memory"""
    return tf.keras.models.load_model("models/best_model.keras")

# Load translator and data loader
translator = Translator()
data_loader = DataLoader("data/train.csv.zip")  # Verify the correct path

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for toxicity prediction"""
    data = request.json
    text = data.get("text", "")

    # Translate text if it's in Portuguese
    text = translator.translate_to_english(text)

    # Load model once
    global model
    if 'model' not in globals():
        model = load_model()

    # Ensure input processing function is applied correctly
    vectorized_input = model.input_processing_function([text])
    prediction = model.predict(np.expand_dims(vectorized_input, 0), batch_size=1).tolist()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))  # Ensure Cloud Run port compatibility
    app.run(host="0.0.0.0", port=port, debug=False)
