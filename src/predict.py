import tensorflow as tf
import numpy as np
import pandas as pd
from data_preprocessing import DataPreprocessor
from model_training import ToxicCommentModel

class Predictor:
    """
    Handles loading the trained model and making predictions on new text inputs.
    """
    def __init__(self, model_path="models/best_model.keras", data_path="data/train.csv.zip"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = self.load_model()
        self.vectorizer = self.load_vectorizer()
    
    def load_model(self):
        """Loads the trained Keras model."""
        print("Loading trained model...")
        return tf.keras.models.load_model(self.model_path)
    
    def load_vectorizer(self):
        """Initializes and fits the vectorizer using the dataset."""
        print("Loading text vectorizer...")
        preprocessor = self.data_path
        df = pd.read_csv(self.data_path)
        X = df['comment_text']
        vectorizer = tf.keras.layers.TextVectorization(max_tokens=200_000, output_sequence_length=1800, output_mode='int')
        vectorizer.adapt(X.values)
        return vectorizer
    
    def predict(self, text):
        """Preprocesses input text and returns model predictions."""
        print("Making prediction...")
        vectorized_text = self.vectorizer(text)  # Convert text to vector format
        prediction = self.model.predict(np.expand_dims(vectorized_text, 0))  # Ensure proper input shape
        return prediction

if __name__ == "__main__":
    predictor = Predictor()
    sample_text = "You freaking suck! I am going to hit you!"
    result = predictor.predict(sample_text)
    print(f"Prediction: {result}")
