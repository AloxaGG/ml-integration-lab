"""Обучение модели классификации Iris и сохранение артефакта."""

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.25
RANDOM_STATE = 42
MAX_ITER = 300
MODEL_PATH = "models/model.pkl"


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    model = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"accuracy={accuracy:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"model_saved={MODEL_PATH}")


if __name__ == "__main__":
    main()
