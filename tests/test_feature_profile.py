"""Модульные тесты профиля признаков и ранней проверки значений.

Проверяют прикладную логику без HTTP: как строится профиль и как функция
`validate_feature_ranges()` описывает нарушения границ.
"""

import numpy as np
import pytest

import config
from src.dataset import check_dataset_matches_config, load_dataset
from src.feature_profile import (
    build_feature_profile,
    describe_violations,
    get_feature_profile,
    validate_feature_ranges,
)

VALID = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_profile_contains_all_features_with_limits():
    profile = get_feature_profile()

    assert list(profile) == list(config.FEATURE_COLUMNS)
    for limits in profile.values():
        assert set(limits) == {"min", "max", "mean"}
        assert limits["min"] <= limits["mean"] <= limits["max"]
        assert all(isinstance(value, float) for value in limits.values())


def test_profile_is_built_from_data():
    data = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])

    profile = build_feature_profile(data)

    assert profile["sepal_length"] == {"min": 1.0, "max": 3.0, "mean": 2.0}
    assert profile["petal_width"] == {"min": 4.0, "max": 6.0, "mean": 5.0}


def test_profile_cache_cannot_be_modified_by_caller():
    """Вызывающий код получает копию: изменение не портит профиль сервиса."""
    get_feature_profile()["sepal_length"]["max"] = 999.0

    assert get_feature_profile()["sepal_length"]["max"] != 999.0


def test_valid_object_has_no_violations():
    assert validate_feature_ranges(VALID) == []


@pytest.mark.parametrize("bound", ["min", "max"], ids=["lower-bound", "upper-bound"])
def test_boundary_values_are_allowed(bound):
    """Границы включительны: значение, равное min или max, допустимо."""
    profile = get_feature_profile()
    features = {name: limits[bound] for name, limits in profile.items()}

    assert validate_feature_ranges(features) == []


def test_out_of_range_value_is_reported_with_limits():
    profile = get_feature_profile()
    features = {**VALID, "petal_width": profile["petal_width"]["max"] + 1}

    violations = validate_feature_ranges(features)

    assert len(violations) == 1
    violation = violations[0]
    assert violation["feature"] == "petal_width"
    assert violation["value"] == profile["petal_width"]["max"] + 1
    assert violation["min"] == profile["petal_width"]["min"]
    assert violation["max"] == profile["petal_width"]["max"]


def test_all_violations_are_returned_at_once():
    features = {name: 100.0 for name in config.FEATURE_COLUMNS}

    violations = validate_feature_ranges(features)

    assert {item["feature"] for item in violations} == set(config.FEATURE_COLUMNS)


def test_message_mentions_feature_value_and_limits():
    violations = validate_feature_ranges({**VALID, "sepal_length": 0.5})

    message = describe_violations(violations)

    assert "sepal_length" in message
    assert "0.5" in message


def test_profile_is_built_from_the_training_dataset():
    """Профиль и обучение берут данные из одного источника и не могут разойтись."""
    features, _ = load_dataset()

    assert get_feature_profile() == build_feature_profile(features)


def test_profile_bounds_cover_every_training_object():
    """Ни один объект обучающего набора не был бы отклонён проверкой."""
    features, _ = load_dataset()
    profile = get_feature_profile()

    for row in features:
        sample = dict(zip(config.FEATURE_COLUMNS, (float(value) for value in row)))
        assert validate_feature_ranges(sample, profile) == []


def test_dataset_matches_declared_interface():
    """Набор согласован с FEATURE_COLUMNS и TARGET_NAMES из config."""
    check_dataset_matches_config()
