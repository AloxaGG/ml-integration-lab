"""Экспорт схемы OpenAPI в файл.

Скрипт импортирует объект FastAPI и вызывает app.openapi(). Веб-сервер при этом
не запускается и обращения к http://127.0.0.1 не выполняются, поэтому схему
можно получить в CI, где сервис не поднят.

Запуск:
    python scripts/export_openapi.py --output site/api/openapi.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api import app  # noqa: E402  (импорт после настройки sys.path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сохраняет схему OpenAPI в файл.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    schema = app.openapi()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"openapi={schema['openapi']} paths={', '.join(schema['paths'])}")
    print(f"saved={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
