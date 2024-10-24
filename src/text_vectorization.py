from tensorflow.keras.layers import TextVectorization

class TextVectorizer:
    """Class to handle text vectorization for NLP tasks."""

    def __init__(self, max_tokens=200_000, output_sequence_length=1800):
        """
        Initializes the TextVectorizer with specified parameters.

        Args:
            max_tokens (int): Maximum number of unique tokens.
            output_sequence_length (int): Length of output sequences.
        """
        self.vectorizer = TextVectorization(max_tokens=max_tokens,
                                             output_sequence_length=output_sequence_length,
                                             output_mode='int')

    def fit(self, texts):
        """Fits the vectorizer on the provided texts."""
        self.vectorizer.adapt(texts)

    def transform(self, texts):
        """Transforms texts into integer sequences."""
        return self.vectorizer(texts)
