"""Единственный источник обучающего набора.

И обучение (`src/train.py`), и профиль признаков (`src/feature_profile.py`)
берут данные только отсюда. Иначе набор можно было бы поменять в одном месте и
не заметить, что второе место осталось на старых данных: сервис начал бы
пропускать значения, которых модель не видела, и отклонять те, на которых она
училась.

Функция `check_dataset_matches_config()` фиксирует эту связь явно — она
сравнивает набор с описанием интерфейса в `src/config.py` и используется в
тестах.
"""

from sklearn.datasets import load_iris

try:  # импорт как часть пакета src (API-слой, тесты)
    from .config import FEATURE_COLUMNS, TARGET_NAMES
except ImportError:  # запуск скриптов из каталога src
    from config import FEATURE_COLUMNS, TARGET_NAMES


def load_dataset():
    """Возвращает (X, y) обучающего набора Iris."""
    data = load_iris()
    return data.data, data.target


def check_dataset_matches_config() -> None:
    """Проверяет, что набор согласован с интерфейсом проекта.

    Поднимает ValueError, если в наборе другое число признаков или классов, чем
    объявлено в `config.FEATURE_COLUMNS` и `config.TARGET_NAMES`.
    """
    features, target = load_dataset()

    if features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"в наборе {features.shape[1]} признаков, "
            f"а config.FEATURE_COLUMNS описывает {len(FEATURE_COLUMNS)}"
        )

    classes = sorted(set(int(value) for value in target))
    if classes != list(range(len(TARGET_NAMES))):
        raise ValueError(
            f"в наборе классы {classes}, "
            f"а config.TARGET_NAMES описывает {len(TARGET_NAMES)}"
        )
