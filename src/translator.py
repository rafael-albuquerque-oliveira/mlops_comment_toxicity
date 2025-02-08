from transformers import pipeline

class Translator:
    """Handles translation of Portuguese text to English"""

    def __init__(self):
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pt-en")

    def translate_to_english(self, text):
        return self.translator(text)[0]['translation_text']
