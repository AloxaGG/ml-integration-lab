"""Модульные тесты прикладного слоя: предсказание без HTTP.

Эти тесты находят ошибку в работе с моделью до того, как она проявится через
API, и не зависят от FastAPI, TestClient и сетевого слоя.
"""

import joblib
import pytest

import config
from src.model_service import ModelNotReady, ModelService

SETOSA = [5.1, 3.5, 1.4, 0.2]
VIRGINICA = [6.7, 3.0, 5.2, 2.3]


def test_predict_one_returns_expected_structure(model_service):
    result = model_service.predict_one(SETOSA)

    assert set(result) == {"prediction", "predicted_name"}
    assert isinstance(result["prediction"], int)
    assert result["prediction"] in (0, 1, 2)
    assert result["predicted_name"] == config.TARGET_NAMES[result["prediction"]]


@pytest.mark.parametrize(
    ("features", "expected_name"),
    [pytest.param(SETOSA, "setosa", id="setosa"), pytest.param(VIRGINICA, "virginica", id="virginica")],
)
def test_predict_one_recognizes_typical_objects(model_service, features, expected_name):
    assert model_service.predict_one(features)["predicted_name"] == expected_name


def test_predict_one_rejects_wrong_number_of_features(model_service):
    with pytest.raises(ValueError):
        model_service.predict_one([5.1, 3.5, 1.4])


def test_missing_artifact_reports_model_not_ready(tmp_path):
    service = ModelService(tmp_path / "no_such_model.pkl")

    assert service.is_ready() is False
    with pytest.raises(ModelNotReady):
        service.predict_one(SETOSA)


def test_model_is_loaded_once(model_service):
    """Повторный вызов не перечитывает артефакт с диска."""
    first = model_service.load()
    second = model_service.load()

    assert first is second


def test_service_uses_saved_artifact(trained_model_path):
    """Интеграционная проверка: сервис предсказывает тем же артефактом, что лежит на диске."""
    service = ModelService(trained_model_path)
    model_from_disk = joblib.load(trained_model_path)

    assert service.predict_one(VIRGINICA)["prediction"] == int(
        model_from_disk.predict([VIRGINICA])[0]
    )
