import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

class ToxicityModel:
    """
    A class to handle text vectorization, dataset preparation, model creation,
    training, and evaluation for a toxicity detection model.
    """

    def __init__(self, max_words=200_000, embedding_dim=32, lstm_units=32, dense_units=[128, 256, 128], output_classes=6):
        """Initialize model parameters and text vectorization."""
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.output_classes = output_classes
        self.vectorizer = TextVectorization(max_tokens=self.max_words, output_sequence_length=1800, output_mode='int')
        self.model = None

    def load_data(self, file_path):
        """Load dataset from a CSV file."""
        df = pd.read_csv(file_path)
        X, y = df['comment_text'], df[df.columns[2:]].values
        return X, y

    def prepare_data(self, X, y, batch_size=16):
        """Vectorize text and create TensorFlow dataset."""
        self.vectorizer.adapt(X.values)
        vectorized_text = self.vectorizer(X.values)
        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y)).cache().shuffle(len(X)).batch(batch_size).prefetch(8)
        train_ds = dataset.take(int(len(dataset) * 0.7))
        val_ds = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
        test_ds = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))
        return train_ds, val_ds, test_ds

    def build_model(self):
        """Build and compile the LSTM-based model."""
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.max_words + 1, output_dim=self.embedding_dim))
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=False)))
        self.model.add(Dropout(0.5))
        for units in self.dense_units:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(0.5))
        self.model.add(Dense(self.output_classes, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
