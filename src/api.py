from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from src.translator import Translator
from src.data_loader import DataLoader  # Ensure correct import path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-pt", torch_dtype=torch.float32)


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Updated model name
model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

data_loader = DataLoader("data/train.csv.zip")  # Verify the correct path


app = Flask(__name__)

translator = Translator()

def load_model():
    """Load the model only when needed to save memory"""
    return tf.keras.models.load_model("models/best_model.keras")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for toxicity prediction"""
    data = request.json
    text = data.get("text", "")

    # Translate text if it's in Portuguese
    text = translator.translate_to_english(text)

    # Load model on demand (lazy loading)
    model = load_model()
    vectorized_input = model.input_processing_function([text])
    prediction = model.predict(np.expand_dims(vectorized_input, 0)).tolist()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))  # Ensure Cloud Run port compatibility
    app.run(host="0.0.0.0", port=port, debug=True)
