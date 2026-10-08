"""Общие фикстуры тестов.

Модель обучается один раз за сессию во временный каталог: тесты не зависят от
наличия `models/model.pkl` в репозитории и не переобучают модель в каждом файле.
Пути импорта задаются в `pytest.ini` (pythonpath), а не правкой sys.path.
"""

import pytest
from fastapi.testclient import TestClient

import train
from app.api import app
from src.model_service import service


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """Обучает модель один раз и возвращает путь к сохраненному артефакту."""
    path = tmp_path_factory.mktemp("models") / "model.pkl"
    model, accuracy = train.train_model()
    assert 0.0 <= accuracy <= 1.0
    train.save_model(model, path)
    return path


@pytest.fixture
def model_service(trained_model_path):
    """Переключает сервис на тестовый артефакт и возвращает его."""
    original_path, original_model = service._path, service._model
    service._path, service._model = trained_model_path, None
    yield service
    service._path, service._model = original_path, original_model


@pytest.fixture
def client(model_service):
    """TestClient с сервисом, настроенным на тестовый артефакт модели."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path):
    """TestClient, у которого артефакт модели отсутствует."""
    original_path, original_model = service._path, service._model
    service._path, service._model = tmp_path / "no_such_model.pkl", None
    with TestClient(app) as test_client:
        yield test_client
    service._path, service._model = original_path, original_model
