from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Translator:
    """
    Translates Brazilian Portuguese text to English using the Hugging Face model "unicamp-dl/translation-pt-en-t5".
    """
    def __init__(self, model_name="unicamp-dl/translation-pt-en-t5"):
        print("Loading translation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def translate(self, text: str) -> str:
        """Translates the given text from Portuguese to English."""
        print("Translating text...")
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model.generate(**inputs)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

if __name__ == "__main__":
    translator = Translator()
    sample_text = "Este é um comentário em português do Brasil."
    translated_text = translator.translate(sample_text)
    print(f"Translated Text: {translated_text}")