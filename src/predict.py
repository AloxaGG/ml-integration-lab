"""Предсказание класса Iris по сохраненной модели."""

import csv

import joblib
from sklearn.datasets import load_iris

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
MODEL_PATH = "models/model.pkl"
SAMPLE_PATH = "data_sample/sample.csv"


def main():
    model = joblib.load(MODEL_PATH)

    with open(SAMPLE_PATH, encoding="utf-8", newline="") as sample_file:
        row = next(csv.DictReader(sample_file))

    features = [[float(row[column]) for column in FEATURE_COLUMNS]]
    predicted_class = int(model.predict(features)[0])
    predicted_name = load_iris().target_names[predicted_class]

    print(f"predicted_class={predicted_class}")
    print(f"predicted_name={predicted_name}")


if __name__ == "__main__":
    main()
