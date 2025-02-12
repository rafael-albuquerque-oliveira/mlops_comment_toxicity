import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from data_preprocessing import DataPreprocessor

class ToxicCommentModel:
    """
    Defines, trains, and saves the LSTM-based model for toxic comment classification.
    """
    def __init__(self, max_words=200_000, embedding_dim=32, lstm_units=32, dense_units=[128, 256, 128], output_classes=6):
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.output_classes = output_classes
        self.model = None
    
    def build_model(self):
        """Builds the LSTM model architecture."""
        print("Building model...")
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.max_words + 1, output_dim=self.embedding_dim))
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=False)))
        self.model.add(Dropout(0.5))
        
        for units in self.dense_units:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(0.5))
        
        self.model.add(Dense(self.output_classes, activation='sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(self.model.summary())
    
    def train_model(self, train_ds, val_ds, model_path="../models/best_model.keras", epochs=15):
        """Trains the model and saves the best version."""
        print("Training model...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, mode='min')
        
        self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
    
if __name__ == "__main__":
    data_path = "../data/train.csv.zip"
    preprocessor = DataPreprocessor(data_path)
    train_ds, val_ds, test_ds, vectorizer = preprocessor.run_preprocessing()
    
    toxic_model = ToxicCommentModel()
    toxic_model.build_model()
    toxic_model.train_model(train_ds, val_ds)