class Config:
    DATA_FILE = '../data/train.csv.zip'
    MAX_WORDS = 200_000
    EMBEDDING_DIM = 32
    LSTM_UNITS = 32
    DENSE_UNITS = [128, 256, 128]
    OUTPUT_CLASSES = 6
    BATCH_SIZE = 16
    EPOCHS = 10
    TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-en-pt"  # Hugging Face model for translation
