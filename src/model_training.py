
# Import necessary libraries
import sys
import os
import pandas as pd

# Add the src directory to the system path to import modules
sys.path.append(os.path.abspath('..'))

from src.data_loader import DataLoader
from src.text_vectorization import TextVectorizer
from src.dataset_preparation import DatasetPreparation
from src.toxicity_pipeline import ToxicityAnalysisPipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Initialize the pipeline with data file path
data_file = '../data/train.csv.zip'
pipeline = ToxicityAnalysisPipeline(data_file)

# Step 2: Run the pipeline to get datasets
train_ds, val_ds, test_ds = pipeline.run()

# Define constants
MAX_WORDS = 200_000  # Adjust this based on your vocabulary size
EMBEDDING_DIM = 32   # Dimension of the embedding layer
LSTM_UNITS = 32      # Number of LSTM units
DENSE_UNITS = [128, 256, 128]  # Units for the fully connected layers
OUTPUT_CLASSES = 6   # Number of output classes

# Initialize the Sequential model
tox_model = Sequential()

# Create the embedding layer
tox_model.add(Embedding(input_dim=MAX_WORDS + 1, output_dim=EMBEDDING_DIM))

# Add Bidirectional LSTM Layer with dropout for regularization
tox_model.add(Bidirectional(LSTM(LSTM_UNITS, activation='tanh', return_sequences=False)))
tox_model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

# Add fully connected layers with dropout for regularization
for units in DENSE_UNITS:
    tox_model.add(Dense(units, activation='relu'))
    tox_model.add(Dropout(0.5))  # Dropout layer after each dense layer

# Final output layer with sigmoid activation for multi-label classification
tox_model.add(Dense(OUTPUT_CLASSES, activation='sigmoid'))

# Compile the model (specify optimizer and loss function)
tox_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
tox_model.summary()

# Define callbacks for better training management
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,          # Stop training if no improvement after 3 epochs
    restore_best_weights=True  # Restore weights from the best epoch
)

model_checkpoint = ModelCheckpoint(
    filepath='best_model.keras',  # Change to .keras extension
    monitor='val_loss',            # Monitor validation loss
    save_best_only=True,           # Save only the best model
    mode='min'                     # Minimize the validation loss
)

# Fit the model with additional options
history = tox_model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[early_stopping, model_checkpoint],  # Include callbacks for better training control
    verbose=1  # Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch
)

import pandas as pd
import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.layers import TextVectorization

df = pd.read_csv('/Users/rafaeloliveira/code/Github Repos/mlops_comment_toxicity/data/train.csv.zip')

X = df['comment_text']
y = df[df.columns[2:]].values

MAX_WORDS = 200_000

vectorizer = TextVectorization(max_tokens=MAX_WORDS,
                               output_sequence_length=1800,
                               output_mode='int')

vectorizer.adapt(X.values)

vectorized_text = vectorizer(X.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160_000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

input_text = vectorizer('You freaking suck! I am going to hit you!')

tox_model.predict(np.expand_dims(input_text, 0))

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test_ds.as_numpy_iterator():
    # Unpack the batch
    X_true, y_true = batch
    # Make prediction
    yhat = tox_model.predict(X_true)

    # Flaatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')
