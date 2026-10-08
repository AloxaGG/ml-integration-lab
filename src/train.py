"""Обучение модели классификации Iris и сохранение артефакта.

Запуск:
    python src/train.py [--model-path PATH]
"""

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import MAX_ITER, MODEL_PATH, RANDOM_STATE, TEST_SIZE


def train_model():
    """Обучает модель и возвращает ее вместе с accuracy на отложенной выборке."""
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
    return model, accuracy


def save_model(model, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели Iris.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"куда сохранить обученную модель (по умолчанию {MODEL_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model, accuracy = train_model()
    print(f"accuracy={accuracy:.3f}")

    save_model(model, args.model_path)
    print(f"model_saved={args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
