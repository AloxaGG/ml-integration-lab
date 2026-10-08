"""Предсказание класса Iris по сохраненной модели.

Запуск:
    python src/predict.py [--model-path PATH] [--input PATH]
"""

import argparse
import csv
from pathlib import Path

import joblib

from config import FEATURE_COLUMNS, MODEL_PATH, SAMPLE_PATH, TARGET_NAMES


class PredictError(Exception):
    """Ошибка, которую можно показать пользователю без трассировки стека."""


def load_model(model_path: Path):
    if not model_path.is_file():
        raise PredictError(
            f"файл модели не найден: {model_path}. "
            "Сначала выполните: python src/train.py"
        )
    return joblib.load(model_path)


def read_features(input_path: Path):
    """Читает первую строку CSV и возвращает список признаков в нужном порядке."""
    if not input_path.is_file():
        raise PredictError(f"файл с входными данными не найден: {input_path}")

    with open(input_path, encoding="utf-8", newline="") as input_file:
        row = next(csv.DictReader(input_file), None)

    if row is None:
        raise PredictError(f"файл {input_path} не содержит строк данных")

    missing = [column for column in FEATURE_COLUMNS if row.get(column) is None]
    if missing:
        raise PredictError(
            f"во входных данных отсутствуют столбцы: {', '.join(missing)}"
        )

    try:
        return [float(row[column]) for column in FEATURE_COLUMNS]
    except ValueError as error:
        raise PredictError(f"нечисловое значение признака: {error}") from error


def predict(model, features):
    """Возвращает (номер класса, имя класса) для одного объекта."""
    predicted_class = int(model.predict([features])[0])
    return predicted_class, TARGET_NAMES[predicted_class]


def parse_args():
    parser = argparse.ArgumentParser(description="Предсказание класса Iris.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"путь к сохраненной модели (по умолчанию {MODEL_PATH})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SAMPLE_PATH,
        dest="input_path",
        help=f"CSV с одним объектом (по умолчанию {SAMPLE_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        model = load_model(args.model_path)
        features = read_features(args.input_path)
    except PredictError as error:
        print(f"error: {error}")
        return 1

    predicted_class, predicted_name = predict(model, features)
    print(f"predicted_class={predicted_class}")
    print(f"predicted_name={predicted_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
