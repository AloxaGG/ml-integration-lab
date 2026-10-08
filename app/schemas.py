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
        json_schema_extra={"examples": [{"prediction": 0, "predicted_name": "setosa"}]}
    )

    prediction: int = Field(description="номер класса Iris: 0, 1 или 2")
    predicted_name: str = Field(description="имя класса: setosa, versicolor, virginica")


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
