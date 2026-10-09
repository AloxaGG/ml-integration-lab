"""Профиль признаков Iris и ранняя проверка значений по его границам.

Профиль — это прозрачный контракт допустимых значений: минимум, максимум и
среднее по каждому признаку обучающего набора. Он вычисляется один раз и
отдаётся клиенту маршрутом `GET /features`, а проверка `validate_feature_ranges()`
выполняется до обращения к модели, чтобы она не получала заведомо невозможные
данные.

Код перенесён из экспериментов `notebooks/source_experiment_updated.ipynb`
(построение профиля) и `notebooks/new_functionality.ipynb` (проверка объекта).

Данные берутся из `src/dataset.py` — из того же места, что и обучение модели,
чтобы профиль не мог разойтись с тем, на чем модель училась.
"""

try:  # импорт как часть пакета src (API-слой, тесты)
    from .config import FEATURE_COLUMNS
    from .dataset import load_dataset
except ImportError:  # запуск скриптов из каталога src
    from config import FEATURE_COLUMNS
    from dataset import load_dataset

# Границы включительны: значение, равное min или max, считается допустимым.
Profile = dict[str, dict[str, float]]

_PROFILE_CACHE: Profile | None = None


def build_feature_profile(data, feature_names=FEATURE_COLUMNS) -> Profile:
    """JSON-совместимые минимум, максимум и среднее для каждого столбца."""
    return {
        name: {
            "min": float(data[:, index].min()),
            "max": float(data[:, index].max()),
            "mean": float(data[:, index].mean()),
        }
        for index, name in enumerate(feature_names)
    }


def get_feature_profile() -> Profile:
    """Профиль признаков: считается один раз и дальше берётся из кеша."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = build_feature_profile(load_dataset()[0])
    # копия, чтобы вызывающий код не мог изменить кеш
    return {name: dict(limits) for name, limits in _PROFILE_CACHE.items()}


def validate_feature_ranges(features: dict, profile: Profile | None = None) -> list[dict]:
    """Возвращает список нарушений границ — сразу все, а не первое найденное.

    Каждое нарушение описывает признак, переданное значение и границы, чтобы
    клиент мог показать понятное сообщение и исправить конкретное поле.
    """
    profile = profile if profile is not None else get_feature_profile()

    violations = []
    for name, limits in profile.items():
        value = float(features[name])
        if not limits["min"] <= value <= limits["max"]:
            violations.append(
                {
                    "feature": name,
                    "value": value,
                    "min": limits["min"],
                    "max": limits["max"],
                }
            )
    return violations


def describe_violations(violations: list[dict]) -> str:
    """Человекочитаемое сообщение по списку нарушений — показывается на странице."""
    parts = [
        f"{item['feature']}={item['value']:g} вне диапазона "
        f"[{item['min']:g}; {item['max']:g}]"
        for item in violations
    ]
    return "Значения вне допустимого диапазона: " + "; ".join(parts)
