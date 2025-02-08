from flask import Flask, request, jsonify
import tensorflow as tf
from model import ToxicityModel
from translator import Translator
import numpy as np

app = Flask(__name__)
translator = Translator()

# Load trained model
tox_model = ToxicityModel()
tox_model.build_model()
tox_model.model.load_weights("best_model.keras")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "")
    translated_text = translator.translate(input_text)
    vectorized_input = tox_model.vectorizer([translated_text])
    prediction = tox_model.model.predict(np.expand_dims(vectorized_input, axis=0)).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
