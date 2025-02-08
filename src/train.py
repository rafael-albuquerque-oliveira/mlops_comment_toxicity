import tensorflow as tf
from src.data_loader import DataLoader
from src.model import ToxicCommentModel

# Load data
data_loader = DataLoader("../data/train.csv.zip")
train_ds, val_ds, test_ds = data_loader.train_ds, data_loader.val_ds, data_loader.test_ds
vectorizer = data_loader.get_vectorizer()

# Build the model
model = ToxicCommentModel().model

# Training callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='../models/best_model.keras', monitor='val_loss', save_best_only=True)

# Train model
model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[early_stopping, model_checkpoint])
