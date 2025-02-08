from model import ToxicityModel

if __name__ == "__main__":
    # Define file path for dataset
    file_path = '../data/train.csv.zip'  # Update with correct path

    # Initialize and execute model pipeline
    tox_model = ToxicityModel()
    X, y = tox_model.load_data(file_path)
    train_ds, val_ds, test_ds = tox_model.prepare_data(X, y)
    tox_model.build_model()
    tox_model.train(train_ds, val_ds)
    tox_model.evaluate(test_ds)
