import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.15


def load_train_test_split(csv_path="fruit_features.csv"):
    """Load fruit_features.csv and return a stratified train/test split, filenames included."""
    data = pd.read_csv(csv_path)
    filenames = data["Filename"]
    X = data.drop(columns=["Filename", "Label"])
    y = data["Label"]
    return train_test_split(X, y, filenames, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
