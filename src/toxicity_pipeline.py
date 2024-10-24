from data_loader import DataLoader
from text_vectorization import TextVectorizer
from dataset_preparation import DatasetPreparation

class ToxicityAnalysisPipeline:
    """Class to manage the entire toxicity analysis pipeline."""

    def __init__(self, data_file):
        """
        Initializes the pipeline with a dataset file path.

        Args:
            data_file (str): Path to the dataset CSV file.
        """
        self.data_file = data_file

    def run(self):
        """Runs the entire pipeline from loading data to preparing datasets."""

        # Step 1: Load Data
        loader = DataLoader(self.data_file)
        X, y = loader.load_data()

        # Step 2: Text Vectorization
        vectorizer = TextVectorizer()
        vectorizer.fit(X.values)  # Fit on text data

        vectorized_text = vectorizer.transform(X.values)

        # Step 3: Prepare Dataset
        prep = DatasetPreparation(vectorized_text, y)

        dataset = prep.create_dataset()  # Create TensorFlow Dataset

        # Step 4: Split Dataset into train/val/test sets
        train_dataset, val_dataset, test_dataset = prep.split_dataset(dataset)

        return train_dataset, val_dataset, test_dataset  # Return datasets for further use
