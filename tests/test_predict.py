"""Smoke-тесты компонента предсказания.

Проверяют, что сохраненная модель загружается, predict.py не падает и выдает
результат ожидаемого формата. Качество модели здесь не проверяется.
"""

import json
import subprocess
import sys

import pytest

import config
import predict
import train


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Обучает модель один раз во временный каталог и возвращает путь к артефакту."""
    path = tmp_path_factory.mktemp("models") / "model.pkl"
    model, accuracy = train.train_model()
    assert 0.0 <= accuracy <= 1.0
    train.save_model(model, path)
    return path


def test_sample_file_exists():
    assert config.SAMPLE_PATH.is_file()


def test_read_features_returns_four_numbers():
    features = predict.read_features(config.SAMPLE_PATH)
    assert len(features) == len(config.FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in features)


def test_model_loads_and_predicts_known_format(model_path):
    model = predict.load_model(model_path)
    features = predict.read_features(config.SAMPLE_PATH)

    predicted_class, predicted_name = predict.predict(model, features)

    assert predicted_class in (0, 1, 2)
    assert predicted_name in config.TARGET_NAMES
    assert predicted_name == config.TARGET_NAMES[predicted_class]


def test_missing_model_reports_readable_error(tmp_path):
    with pytest.raises(predict.PredictError):
        predict.load_model(tmp_path / "no_such_model.pkl")


def test_missing_input_file_reports_readable_error(tmp_path):
    with pytest.raises(predict.PredictError):
        predict.read_features(tmp_path / "no_such_sample.csv")


def test_bad_input_file_reports_readable_error(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("sepal_length,sepal_width\n1.0,2.0\n", encoding="utf-8")
    with pytest.raises(predict.PredictError):
        predict.read_features(bad_csv)


def test_format_prediction_text_is_default_format():
    assert config.DEFAULT_OUTPUT_FORMAT == "text"
    output = predict.format_prediction(0, "setosa", "text")
    assert output.splitlines() == ["predicted_class=0", "predicted_name=setosa"]


def test_format_prediction_json():
    output = predict.format_prediction(2, "virginica", "json")
    assert json.loads(output) == {"predicted_class": 2, "predicted_name": "virginica"}


def run_predict_script(*args):
    """Запускает predict.py отдельным процессом не из корня репозитория."""
    return subprocess.run(
        [sys.executable, str(config.PROJECT_ROOT / "src" / "predict.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=config.PROJECT_ROOT.parent,
    )


def test_predict_script_runs_end_to_end(model_path):
    """predict.py запускается как отдельный процесс и печатает ожидаемые строки."""
    result = run_predict_script("--model-path", str(model_path))

    assert result.returncode == 0, result.stderr
    assert "predicted_class=" in result.stdout
    assert "predicted_name=" in result.stdout


def test_predict_script_json_output(model_path):
    """С --format json вывод целиком разбирается как один JSON-объект."""
    result = run_predict_script("--model-path", str(model_path), "--format", "json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["predicted_class"] in (0, 1, 2)
    assert payload["predicted_name"] == config.TARGET_NAMES[payload["predicted_class"]]


def test_predict_script_json_error_goes_to_stderr(tmp_path):
    """В режиме json ошибка тоже JSON, а stdout остается пустым."""
    missing_model = tmp_path / "no_such_model.pkl"
    result = run_predict_script("--model-path", str(missing_model), "--format", "json")

    assert result.returncode == 1
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert "no_such_model.pkl" in payload["error"]


def test_predict_script_rejects_unknown_format(model_path):
    result = run_predict_script("--model-path", str(model_path), "--format", "xml")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
