"""FastAPI-приложение: HTTP-интерфейс к сохраненной модели.

Запуск:
    python -m uvicorn app.api:app --reload

Маршруты:
    GET  /         — клиентская страница из static/index.html;
    GET  /health   — состояние сервиса и готовность модели;
    GET  /features — профиль признаков: допустимые диапазоны и средние значения;
    POST /predict  — проверка объекта по профилю и предсказание класса Iris;
    GET  /docs, /openapi.json — документация, которую FastAPI строит по схемам.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    ErrorResponse,
    FeatureLimits,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ValidationErrorResponse,
)
from src.config import PROJECT_ROOT
from src.feature_profile import (
    describe_violations,
    get_feature_profile,
    validate_feature_ranges,
)
from src.model_service import ModelNotReady, service

STATIC_DIR = PROJECT_ROOT / "static"
INDEX_PAGE = STATIC_DIR / "index.html"

DESCRIPTION = """
Учебный сервис классификации ирисов.

Модель обучается отдельно командой `python src/train.py` и сохраняется в
`models/model.pkl`. Сервис только загружает готовый артефакт и выполняет
предсказание — обучение внутри запроса не выполняется.

Перед предсказанием значения признаков проверяются по профилю обучающего
набора (`GET /features`): объект за пределами известных диапазонов
отклоняется с кодом 422 и не доходит до модели.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Готовит сервис к работе: модель и профиль признаков считаются один раз.

    Если артефакта модели нет, сервис все равно поднимается: /health покажет
    model_ready=false, а /predict вернет 503 с понятным сообщением. Профиль
    признаков от модели не зависит, поэтому /features работает всегда.
    """
    get_feature_profile()
    try:
        service.load()
    except ModelNotReady:
        pass
    yield


app = FastAPI(
    title="ML Integration API",
    description=DESCRIPTION,
    version="1.1.0",
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Единый формат ответа 422 для ошибок Pydantic и проверки по профилю.

    Клиент получает читаемое поле `detail` для показа пользователю и
    машиночитаемый список `errors` с подробностями по каждому полю.
    """
    errors = exc.errors()
    messages = []
    for error in errors:
        field = ".".join(str(part) for part in error["loc"][1:]) or "тело запроса"
        messages.append(f"{field}: {error['msg']}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Некорректный запрос: " + "; ".join(messages),
            "errors": jsonable_errors(errors),
        },
    )


def jsonable_errors(errors: list) -> list:
    """Убирает из ошибок Pydantic несериализуемые значения (например, исключения)."""
    cleaned = []
    for error in errors:
        item = {key: value for key, value in error.items() if key != "ctx"}
        item["loc"] = list(item.get("loc", []))
        cleaned.append(item)
    return cleaned


@app.get(
    "/",
    summary="Клиентская страница",
    tags=["ui"],
    response_class=FileResponse,
    responses={200: {"content": {"text/html": {}}, "description": "HTML-страница"}},
)
def index() -> FileResponse:
    """Отдает страницу проверки признаков с того же приложения."""
    if not INDEX_PAGE.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"страница не найдена: {INDEX_PAGE}",
        )
    return FileResponse(INDEX_PAGE, media_type="text/html")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Состояние сервиса",
    tags=["service"],
)
def health() -> HealthResponse:
    """Возвращает состояние сервиса и признак готовности модели."""
    return HealthResponse(status="ok", model_ready=service.is_ready())


@app.get(
    "/features",
    response_model=dict[str, FeatureLimits],
    summary="Профиль признаков",
    tags=["service"],
)
def features() -> dict:
    """Допустимые границы и средние значения четырех признаков Iris.

    Формат пригоден фронтенду без преобразования: ключ — имя признака,
    значение — объект с полями min, max и mean.
    """
    return get_feature_profile()


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Проверка по профилю и предсказание класса Iris",
    tags=["model"],
    responses={
        422: {
            "model": ValidationErrorResponse,
            "description": "Значение вне допустимого диапазона или неверный запрос",
        },
        503: {"model": ErrorResponse, "description": "Модель недоступна"},
    },
)
def predict(request: PredictRequest):
    """Проверяет объект по профилю признаков и предсказывает класс.

    Значения за границами профиля отклоняются до обращения к модели: сервис
    возвращает 422 с именем признака, переданным значением и границами.
    """
    violations = validate_feature_ranges(request.model_dump())
    if violations:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": describe_violations(violations), "errors": violations},
        )

    try:
        result = service.predict_one(request.to_features())
    except ModelNotReady as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return PredictResponse(**result)
