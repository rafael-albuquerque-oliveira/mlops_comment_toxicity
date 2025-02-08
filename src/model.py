import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

class ToxicCommentModel:
    """Builds and compiles the LSTM model for toxicity classification"""

    def __init__(self, max_words=200_000, embedding_dim=32, lstm_units=32, dense_units=[128, 256, 128], output_classes=6):
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.output_classes = output_classes
        self.model = self.build_model()

    def build_model(self):
        """Constructs the model"""
        model = Sequential([
            Embedding(input_dim=self.max_words + 1, output_dim=self.embedding_dim),
            Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=False)),
            Dropout(0.5)
        ])
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.5))
        model.add(Dense(self.output_classes, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
