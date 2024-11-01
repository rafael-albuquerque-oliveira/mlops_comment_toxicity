from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        """Build and compile the Keras model."""
        model = Sequential()
        model.add(Embedding(input_dim=self.config.MAX_WORDS + 1,
                            output_dim=self.config.EMBEDDING_DIM))

        model.add(Bidirectional(LSTM(self.config.LSTM_UNITS,
                                     activation='tanh',
                                     return_sequences=False)))
        model.add(Dropout(0.5))

        for units in self.config.DENSE_UNITS:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.5))

        model.add(Dense(self.config.OUTPUT_CLASSES, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, train_ds, val_ds):
        """Train the Keras model with callbacks."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=3,
                                       restore_best_weights=True)

        model_checkpoint = ModelCheckpoint(filepath='best_model.keras',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode='min')

        history = self.model.fit(train_ds,
                                 epochs=self.config.EPOCHS,
                                 validation_data=val_ds,
                                 callbacks=[early_stopping, model_checkpoint],
                                 verbose=1)

        return history
