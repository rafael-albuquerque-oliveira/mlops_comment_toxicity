from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from src.translator import Translator

app = Flask(__name__)

# Load model and vectorizer
model = tf.keras.models.load_model('../models/best_model.keras')
data_loader = DataLoader("../data/train.csv.zip")
vectorizer = data_loader.get_vectorizer()
translator = Translator()

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for toxicity prediction"""
    data = request.json
    text = data.get("text", "")

    # Translate text if it's in Portuguese
    text = translator.translate_to_english(text)

    vectorized_input = vectorizer([text])
    prediction = model.predict(np.expand_dims(vectorized_input, 0)).tolist()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
