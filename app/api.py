"""FastAPI-приложение: HTTP-интерфейс к сохраненной модели.

Запуск:
    python -m uvicorn app.api:app --reload

Маршруты:
    GET  /health  — состояние сервиса и готовность модели;
    POST /predict — предсказание класса Iris по признакам одного объекта;
    GET  /docs, /openapi.json — документация, которую FastAPI строит по схемам.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse
from src.model_service import ModelNotReady, service

DESCRIPTION = """
Учебный сервис классификации ирисов.

Модель обучается отдельно командой `python src/train.py` и сохраняется в
`models/model.pkl`. Сервис только загружает готовый артефакт и выполняет
предсказание — обучение внутри запроса не выполняется.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель один раз при старте приложения.

    Если артефакта нет, сервис все равно поднимается: /health покажет
    model_ready=false, а /predict вернет 503 с понятным сообщением.
    """
    try:
        service.load()
    except ModelNotReady:
        pass
    yield


app = FastAPI(
    title="ML Integration API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Состояние сервиса",
    tags=["service"],
)
def health() -> HealthResponse:
    """Возвращает состояние сервиса и признак готовности модели."""
    return HealthResponse(status="ok", model_ready=service.is_ready())


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Предсказание класса Iris",
    tags=["model"],
    responses={503: {"model": ErrorResponse, "description": "Модель недоступна"}},
)
def predict(request: PredictRequest) -> PredictResponse:
    """Предсказывает класс для одного объекта по четырем признакам."""
    try:
        result = service.predict_one(request.to_features())
    except ModelNotReady as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return PredictResponse(**result)
