import pandas as pd

class DataLoader:
    """Class to load data from CSV files."""

    def __init__(self, file_path):
        """
        Initializes the DataLoader with the specified file path.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
        """
        self.file_path = file_path

    def load_data(self):
        """Loads data from the specified CSV file and returns features and labels."""
        df = pd.read_csv(self.file_path)
        X = df['comment_text']
        y = df[df.columns[2:]].values  # Adjust based on your label columns
        return X, y
