"""Схемы запроса и ответа API.

Pydantic-модели задают контракт сервиса: по ним FastAPI проверяет входной JSON
и формирует описание в OpenAPI. Имена полей совпадают с признаками, которые
принимает модель (`src/config.py`, FEATURE_COLUMNS).
"""

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Признаки одного цветка ириса (все значения в сантиметрах, > 0)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    )

    sepal_length: float = Field(gt=0, description="длина чашелистика, см")
    sepal_width: float = Field(gt=0, description="ширина чашелистика, см")
    petal_length: float = Field(gt=0, description="длина лепестка, см")
    petal_width: float = Field(gt=0, description="ширина лепестка, см")

    def to_features(self) -> list[float]:
        """Признаки в том порядке, в котором их ожидает модель."""
        return [
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]


class PredictResponse(BaseModel):
    """Результат предсказания: номер класса и его человекочитаемое имя."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"class_id": 0, "class_name": "setosa"}]}
    )

    class_id: int = Field(description="номер класса Iris: 0, 1 или 2")
    class_name: str = Field(description="имя класса: setosa, versicolor, virginica")


class FeatureLimits(BaseModel):
    """Допустимый диапазон одного признака и его обычное значение."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"min": 4.3, "max": 7.9, "mean": 5.84}]}
    )

    min: float = Field(description="минимальное допустимое значение (включительно)")
    max: float = Field(description="максимальное допустимое значение (включительно)")
    mean: float = Field(description="среднее значение по обучающему набору")


class RangeViolation(BaseModel):
    """Одно нарушение границ признака."""

    feature: str = Field(description="имя признака")
    value: float = Field(description="переданное значение")
    min: float = Field(description="нижняя граница")
    max: float = Field(description="верхняя граница")


class ValidationErrorResponse(BaseModel):
    """Ответ 422: читаемое сообщение и машиночитаемый список ошибок."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": "Значения вне допустимого диапазона: "
                    "petal_width=4 вне диапазона [0.1; 2.5]",
                    "errors": [
                        {"feature": "petal_width", "value": 4.0, "min": 0.1, "max": 2.5}
                    ],
                }
            ]
        }
    )

    detail: str = Field(description="сообщение для пользователя")
    errors: list = Field(description="подробности: по одному элементу на ошибку")


class HealthResponse(BaseModel):
    """Состояние сервиса и готовность модели."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "ok", "model_ready": True}]}
    )

    status: str = Field(description="общее состояние сервиса")
    model_ready: bool = Field(description="артефакт модели загружен, /predict доступен")


class ErrorResponse(BaseModel):
    """Тело ответа при ошибке."""

    detail: str
