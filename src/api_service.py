import bentoml
from bentoml.io import Text, JSON
from src.data_loader import DataLoader
from src.model import ToxicityModel
from transformers import pipeline

# Load the translation model
translator = pipeline("translation_pt_to_en", model="Helsinki-NLP/opus-mt-pt-en")

# Load the toxicity model
tox_model = ToxicityModel(max_words=200_000)
tox_model.model.load_weights('best_model.keras')

# Create a BentoML service
svc = bentoml.Service("toxicity_classifier")

@svc.api(input=Text(), output=JSON())
def classify(text: str):
    """Classify the toxicity of a Portuguese text."""
    # Translate to English
    translated_text = translator(text, max_length=400)[0]['translation_text']
    # Vectorize the text
    vectorized_text = DataLoader.vectorizer([translated_text])
    # Predict toxicity
    prediction = tox_model.model.predict(vectorized_text)
    return {"prediction": prediction.tolist()}
