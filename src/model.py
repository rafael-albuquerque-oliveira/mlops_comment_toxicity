from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ToxicityModel:
    def __init__(self, max_words, embedding_dim=32, lstm_units=32, dense_units=[128, 256, 128], output_classes=6):
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.output_classes = output_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build the LSTM-based toxicity classification model."""
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words + 1, output_dim=self.embedding_dim))
        model.add(Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=False)))
        model.add(Dropout(0.5))
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.5))
        model.add(Dense(self.output_classes, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_ds, val_ds, epochs=10):
        """Train the model with early stopping and model checkpointing."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
        history = self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[early_stopping, model_checkpoint])
        return history
