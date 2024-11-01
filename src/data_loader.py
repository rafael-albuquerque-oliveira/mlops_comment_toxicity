import pandas as pd

class DataLoader:
    def __init__(self, data_file):
        self.data_file = data_file

    def load_data(self):
        """Load dataset from CSV file."""
        df = pd.read_csv(self.data_file)
        X = df['comment_text']
        y = df[df.columns[2:]].values
        return X, y
