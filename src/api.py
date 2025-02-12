from flask import Flask, request, jsonify
from predict import Predictor
from translator import Translator

app = Flask(__name__)

# Load model and translator
predictor = Predictor()
translator = Translator()

LABELS = ["Tóxico", "Severamente Tóxico", "Obsceno", "Ameaça", "Insulto", "Ódio Identitário"]

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to receive a comment, translate it if necessary, and return toxicity predictions."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    
    # Translate if input is in Portuguese
    translated_text = translator.translate(text)
    prediction = predictor.predict(translated_text)
    
    # Format response
    response = {
        "original_text": text,
        "translated_text": translated_text,
        "prediction": dict(zip(LABELS, prediction[0].tolist()))
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)