import tensorflow as tf

class TextVectorizer:
    def __init__(self, max_tokens):
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                            output_sequence_length=1800,
                                                            output_mode='int')

    def fit(self, texts):
        """Fit the vectorizer to the texts."""
        self.vectorizer.adapt(texts)

    def transform(self, texts):
        """Transform texts to vectorized form."""
        return self.vectorizer(texts)
