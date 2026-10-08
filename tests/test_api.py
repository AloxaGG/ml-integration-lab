"""Тесты HTTP-слоя через FastAPI TestClient.

TestClient поднимает приложение внутри процесса pytest, поэтому отдельный
uvicorn и браузер не нужны: запросы идут напрямую в ASGI-приложение.
"""

import pytest

VALID_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_predict_returns_prediction(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "predicted_name"}
    assert isinstance(body["prediction"], int)
    assert body["prediction"] in (0, 1, 2)
    assert body["predicted_name"] == "setosa"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {k: v for k, v in VALID_PAYLOAD.items() if k != "petal_width"},
            id="missing-field",
        ),
        pytest.param({**VALID_PAYLOAD, "sepal_length": "wide"}, id="wrong-type"),
        pytest.param({**VALID_PAYLOAD, "petal_width": 0}, id="not-positive"),
        pytest.param({}, id="empty-body"),
    ],
)
def test_predict_rejects_invalid_payload(client, payload):
    """Pydantic отклоняет некорректный JSON статусом 422, приложение не падает."""
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_validation_error_describes_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "petal_width"}

    detail = client.post("/predict", json=payload).json()["detail"]

    assert any("petal_width" in error["loc"] for error in detail)


def test_service_stays_alive_after_invalid_request(client):
    client.post("/predict", json={"sepal_length": "abc"})

    assert client.get("/health").status_code == 200


def test_predict_without_model_returns_503(client_without_model):
    response = client_without_model.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 503
    assert "модели" in response.json()["detail"]


def test_health_reports_model_not_ready(client_without_model):
    body = client_without_model.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_ready"] is False


def test_openapi_describes_contract(client):
    schema = client.get("/openapi.json").json()

    assert schema["openapi"].startswith("3.")
    assert {"/health", "/predict"} <= set(schema["paths"])
    assert set(schema["components"]["schemas"]["PredictRequest"]["properties"]) == set(
        VALID_PAYLOAD
    )


def test_docs_page_is_available(client):
    assert client.get("/docs").status_code == 200
