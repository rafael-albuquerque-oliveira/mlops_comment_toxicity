from .data_loader import DataLoader
from .text_vectorization import TextVectorizer
from .dataset_preparation import DatasetPreparation
from .model_trainer import ModelTrainer

class ToxicityAnalysisPipeline:
    def __init__(self, data_file):
        self.data_file = data_file

    def run(self):
        """Run the entire pipeline for toxicity analysis."""
        # Load data
        data_loader = DataLoader(self.data_file)
        X, y = data_loader.load_data()

        # Vectorize text data
        vectorizer = TextVectorizer(max_tokens=200_000)
        vectorizer.fit(X.values)
        vectorized_texts = vectorizer.transform(X.values)

        # Prepare datasets for training and validation
        dataset_prep = DatasetPreparation(vectorized_texts.numpy(), y)
        train_ds, val_ds, test_ds = dataset_prep.create_dataset()

        # Train the model using the prepared datasets
        trainer = ModelTrainer(config)
        trainer.train(train_ds, val_ds)
