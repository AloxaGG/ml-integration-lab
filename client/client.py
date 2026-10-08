"""Клиент учебного стенда: отправляет один запрос к API и сохраняет ответ.

Адрес API и путь к результату берутся из переменных окружения, потому что внутри
Compose клиент обращается к сервису по имени (`http://api:8000`), а не к
localhost: localhost внутри контейнера client указывает на сам client.
"""

import json
import os
import sys
from pathlib import Path

import requests

API_URL = os.getenv("API_URL", "http://api:8000")
RESULT_PATH = Path(os.getenv("RESULT_PATH", "/results/prediction.json"))
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))

PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def main() -> int:
    print(f"API_URL={API_URL}")
    print(f"POST {API_URL}/predict payload={PAYLOAD}")

    try:
        response = requests.post(f"{API_URL}/predict", json=PAYLOAD, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"error: запрос к API не выполнен: {error}", file=sys.stderr)
        return 1

    result = response.json()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"HTTP {response.status_code}, ответ сохранен в {RESULT_PATH}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
