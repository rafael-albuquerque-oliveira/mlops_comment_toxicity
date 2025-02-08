import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

class DataLoader:
    def __init__(self, data_path, max_words=200_000, sequence_length=1800):
        self.data_path = data_path
        self.max_words = max_words
        self.sequence_length = sequence_length
        self.vectorizer = TextVectorization(
            max_tokens=max_words,
            output_sequence_length=sequence_length,
            output_mode='int'
        )

    def load_data(self):
        """Load and preprocess the dataset."""
        df = pd.read_csv(self.data_path)
        X = df['comment_text']
        y = df[df.columns[2:]].values
        return X, y

    def vectorize_text(self, X):
        """Adapt and vectorize the text data."""
        self.vectorizer.adapt(X.values)
        return self.vectorizer(X.values)

    def create_dataset(self, vectorized_text, y):
        """Create a TensorFlow dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        dataset = dataset.cache().shuffle(160_000).batch(16).prefetch(8)
        return dataset

    def split_dataset(self, dataset):
        """Split the dataset into train, validation, and test sets."""
        train_ds = dataset.take(int(len(dataset) * 0.7))
        val_ds = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
        test_ds = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))
        return train_ds, val_ds, test_ds
