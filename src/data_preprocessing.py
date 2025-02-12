import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.layers import TextVectorization

class DataPreprocessor:
    """
    Handles data loading and preprocessing for the Toxic Comment Classification project.
    """
    def __init__(self, data_path: str, max_words: int = 200_000, seq_length: int = 1800, batch_size: int = 16):
        self.data_path = data_path
        self.max_words = max_words
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vectorizer = TextVectorization(max_tokens=self.max_words, 
                                            output_sequence_length=self.seq_length,
                                            output_mode='int')
        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load_data(self):
        """Loads dataset from the given path."""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        X = df['comment_text']
        y = df[df.columns[2:]].values
        return X, y

    def prepare_vectorizer(self, X):
        """Fits the TextVectorization layer to the input data."""
        print("Fitting text vectorizer...")
        self.vectorizer.adapt(X.values)
        return self.vectorizer(X.values)

    def create_dataset(self, X, y):
        """Creates a TensorFlow dataset pipeline."""
        print("Creating TensorFlow dataset...")
        vectorized_text = self.prepare_vectorizer(X)
        self.dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        self.dataset = self.dataset.cache().shuffle(len(X)).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2):
        """Splits dataset into train, validation, and test sets."""
        print("Splitting dataset...")
        train_size = int(len(self.dataset) * train_ratio)
        val_size = int(len(self.dataset) * val_ratio)
        
        self.train_ds = self.dataset.take(train_size)
        self.val_ds = self.dataset.skip(train_size).take(val_size)
        self.test_ds = self.dataset.skip(train_size + val_size)

    def run_preprocessing(self):
        """Executes the full data preprocessing pipeline."""
        X, y = self.load_data()
        self.create_dataset(X, y)
        self.split_dataset()
        print("Data preprocessing completed!")
        return self.train_ds, self.val_ds, self.test_ds, self.vectorizer

if __name__ == "__main__":
    data_path = "../data/train.csv.zip"  # Adjust based on your directory structure
    preprocessor = DataPreprocessor(data_path)
    train_ds, val_ds, test_ds, vectorizer = preprocessor.run_preprocessing()