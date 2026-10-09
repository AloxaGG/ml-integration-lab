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
    assert set(body) == {"class_id", "class_name"}
    assert isinstance(body["class_id"], int)
    assert body["class_id"] in (0, 1, 2)
    assert body["class_name"] == "setosa"


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

    body = client.post("/predict", json=payload).json()

    assert "petal_width" in body["detail"]
    assert any("petal_width" in error["loc"] for error in body["errors"])


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


def test_features_returns_profile_for_all_fields(client):
    """GET /features отдает профиль в формате, пригодном фронтенду без преобразования."""
    response = client.get("/features")

    assert response.status_code == 200
    profile = response.json()
    assert set(profile) == set(VALID_PAYLOAD)
    for limits in profile.values():
        assert set(limits) == {"min", "max", "mean"}
        assert limits["min"] <= limits["mean"] <= limits["max"]


def test_predict_accepts_boundary_values(client):
    """Границы профиля включительны: объект на границе проходит проверку."""
    profile = client.get("/features").json()
    payload = {name: limits["max"] for name, limits in profile.items()}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["class_id"] in (0, 1, 2)


def test_predict_rejects_value_out_of_profile_range(client):
    """Значение вне диапазона отклоняется до обращения к модели."""
    profile = client.get("/features").json()
    too_large = profile["petal_width"]["max"] + 2
    payload = {**VALID_PAYLOAD, "petal_width": too_large}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "petal_width" in body["detail"]
    violation = body["errors"][0]
    assert violation["feature"] == "petal_width"
    assert violation["value"] == too_large
    assert violation["min"] == profile["petal_width"]["min"]
    assert violation["max"] == profile["petal_width"]["max"]


def test_out_of_range_request_does_not_reach_the_model(client, model_service):
    """Отклоненный объект не доходит до модели: предсказание не выполняется."""
    calls = []
    original_predict_one = model_service.predict_one
    model_service.predict_one = lambda features: calls.append(features) or original_predict_one(features)
    try:
        client.post("/predict", json={**VALID_PAYLOAD, "sepal_length": 99.0})
        assert calls == []
        client.post("/predict", json=VALID_PAYLOAD)
        assert len(calls) == 1
    finally:
        model_service.predict_one = original_predict_one


def test_index_page_is_served_as_html(client):
    """GET / отдает клиентскую страницу с того же приложения."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<form id=\"form\">" in response.text
    assert "/features" in response.text
