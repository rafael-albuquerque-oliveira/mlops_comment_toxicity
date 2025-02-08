from transformers import pipeline

class Translator:
    """A class to translate Portuguese text to English using Hugging Face models."""

    def __init__(self):
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-pt-en")

    def translate(self, text):
        """Translate Portuguese text to English."""
        return self.translator(text)[0]['translation_text']
