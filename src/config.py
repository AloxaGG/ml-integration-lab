"""Пути и параметры проекта.

Все пути вычисляются от корня репозитория, а не от текущего рабочего каталога,
поэтому скрипты дают одинаковый результат при запуске из любого места.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SAMPLE_PATH = PROJECT_ROOT / "data_sample" / "sample.csv"

# Признаки входного объекта: порядок столбцов задает интерфейс компонента
# и совпадает с порядком признаков в наборе Iris.
FEATURE_COLUMNS = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)

# Гиперпараметры обучения
TEST_SIZE = 0.25
RANDOM_STATE = 42
MAX_ITER = 300

# Имена классов Iris в порядке меток 0, 1, 2
TARGET_NAMES = ("setosa", "versicolor", "virginica")
