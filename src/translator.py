from transformers import pipeline

class Translator:
    def __init__(self):
        """Initialize the Hugging Face translation pipeline."""
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-pt")

    def translate(self, text):
        """Translate text from English to Brazilian Portuguese."""
        translated_texts = self.translator(text, max_length=400)

        # Return only the first translation result.
        return translated_texts[0]['translation_text']
