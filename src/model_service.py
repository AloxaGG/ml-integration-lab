"""Прикладной слой предсказания: загрузка артефакта модели и предсказание.

Модуль не знает ничего про HTTP и про обучение модели. Его используют и
CLI-скрипт `src/predict.py`, и API-слой `app/api.py`, поэтому логика работы с
моделью описана один раз и одинаково ведет себя в обоих сценариях.

Путь к модели берется из переменной окружения MODEL_PATH, а если она не задана —
из `src/config.py`. Это нужно для запуска в контейнере, где артефакт лежит по
пути /app/models/model.pkl.
"""

import os
from pathlib import Path

import joblib

try:  # импорт как часть пакета src (API-слой, тесты)
    from .config import FEATURE_COLUMNS, MODEL_PATH, TARGET_NAMES
except ImportError:  # запуск скриптов из каталога src (практики 1-2)
    from config import FEATURE_COLUMNS, MODEL_PATH, TARGET_NAMES


class ModelNotReady(RuntimeError):
    """Артефакт модели недоступен, предсказание выполнить нельзя."""


def model_path() -> Path:
    """Путь к артефакту модели: MODEL_PATH из окружения или значение по умолчанию."""
    env_path = os.getenv("MODEL_PATH")
    return Path(env_path) if env_path else MODEL_PATH


class ModelService:
    """Держит загруженную модель в памяти: артефакт читается с диска один раз."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._model = None

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else model_path()

    def load(self):
        """Загружает модель с диска (повторные вызовы возвращают кеш)."""
        if self._model is None:
            path = self.path
            if not path.is_file():
                raise ModelNotReady(
                    f"файл модели не найден: {path}. "
                    "Сначала выполните: python src/train.py"
                )
            self._model = joblib.load(path)
        return self._model

    def is_ready(self) -> bool:
        """True, если модель уже загружена или артефакт есть на диске."""
        return self._model is not None or self.path.is_file()

    def predict_one(self, features) -> dict:
        """Предсказание для одного объекта: список из 4 признаков -> dict.

        Возвращает {"class_id": int, "class_name": str} — те же поля, что отдает
        маршрут POST /predict.
        """
        features = list(features)
        if len(features) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"ожидается {len(FEATURE_COLUMNS)} признака, получено {len(features)}"
            )

        model = self.load()
        predicted_class = int(model.predict([features])[0])
        return {
            "class_id": predicted_class,
            "class_name": TARGET_NAMES[predicted_class],
        }


# Единственный экземпляр на процесс: модель не перечитывается на каждый запрос.
service = ModelService()


def predict_one(features) -> dict:
    return service.predict_one(features)
