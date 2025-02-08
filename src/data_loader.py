import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization

class DataLoader:
    """Handles data loading, preprocessing, and vectorization for training"""

    def __init__(self, file_path, max_words=200_000, sequence_length=1800, batch_size=16):
        self.file_path = file_path
        self.max_words = max_words
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vectorizer = None
        self.train_ds, self.val_ds, self.test_ds = self.load_data()

    def load_data(self):
        """Loads and processes the dataset"""
        df = pd.read_csv(self.file_path)
        X, y = df['comment_text'].values, df[df.columns[2:]].values

        self.vectorizer = TextVectorization(max_tokens=self.max_words,
                                            output_sequence_length=self.sequence_length,
                                            output_mode='int')
        self.vectorizer.adapt(X)

        vectorized_text = self.vectorizer(X)
        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        dataset = dataset.shuffle(len(df)).batch(self.batch_size).prefetch(8)

        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size).take(val_size)
        test_ds = dataset.skip(train_size + val_size)

        return train_ds, val_ds, test_ds

    def get_vectorizer(self):
        """Returns the text vectorizer"""
        return self.vectorizer
