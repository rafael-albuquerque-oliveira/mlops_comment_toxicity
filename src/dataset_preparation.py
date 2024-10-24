import tensorflow as tf

class DatasetPreparation:
    """Class to prepare TensorFlow datasets from vectorized text and labels."""

    def __init__(self, vectorized_text, labels):
        """
        Initializes the DatasetPreparation with vectorized text and labels.

        Args:
            vectorized_text: Vectorized text data.
            labels: Corresponding labels for the text data.
        """
        self.vectorized_text = vectorized_text
        self.labels = labels

    def create_dataset(self):
        """Creates a TensorFlow dataset from the vectorized text and labels."""
        dataset = tf.data.Dataset.from_tensor_slices((self.vectorized_text, self.labels))

        # Cache, shuffle, batch, and prefetch for performance optimization
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=160_000)
        dataset = dataset.batch(16)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def split_dataset(self, dataset):
        """Splits the dataset into training, validation, and test sets."""

        train_size = int(len(dataset) * 0.7)
        val_size = int(len(dataset) * 0.2)

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)

        return train_dataset, val_dataset, test_dataset
