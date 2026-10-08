# ML Integration Lab 1

Учебный проект практической работы 1 по дисциплине «Разработка и интеграция»
(магистратура 01.04.02).

## Цель проекта

Преобразовать исследовательский ML-код (notebook) в минимальный инженерный
проект, пригодный для дальнейшей интеграции в программную систему.

Прикладная задача модели — классификация ирисов (набор Iris, 4 числовых
признака, 3 класса) с помощью логистической регрессии. Проект разделяет два
независимых сценария:

- **обучение** — `src/train.py` обучает модель и сохраняет артефакт на диск;
- **инференс** — `src/predict.py` загружает *уже сохраненный* артефакт и
  предсказывает класс для одного объекта из CSV.

## Исходная заготовка

Исходником послужила выданная преподавателем заготовка `starter_ml.ipynb`
(получена в виде ZIP-архива вместе с `sample.csv` и `requirements-starter.txt`).
В ней загрузка данных, обучение, сохранение модели и предсказание находились в
одном notebook.

Перед добавлением в репозиторий notebook был проверен и очищен: удалены
результаты выполнения ячеек и счетчики запусков. Секретов, персональных данных
и абсолютных локальных путей в исходнике не обнаружено, поэтому копия
безопасна и хранится в Git как `notebooks/source_experiment.ipynb`. Этот файл
сохранен как исторический исходник и **не используется** рабочим кодом.

Обнаруженные в заготовке инженерные недостатки и что с ними сделано:

| Недостаток исходника | Решение в проекте |
| --- | --- |
| Обучение и инференс в одном файле | Разделены на `src/train.py` и `src/predict.py` |
| Пути к `model.pkl` и `sample.csv` зависят от текущего рабочего каталога (запуск не из корня падал с `FileNotFoundError`) | Пути вычисляются от корня репозитория в `src/config.py` |
| Входной формат не оформлен как интерфейс | Список и порядок признаков зафиксированы в `config.FEATURE_COLUMNS`, столбцы проверяются при чтении |
| Нет обработки ошибок | Понятные сообщения и код возврата 1 вместо трассировки стека |
| Нет тестов и документации | Добавлены `tests/test_predict.py` и этот README |
| Временное имя файла зависимостей | Итоговый `requirements.txt` |

## Структура проекта

```text
ml-integration-lab/
├── README.md                          # документация проекта
├── .gitignore                         # окружения, кэши, артефакты, файлы IDE
├── .dockerignore                      # что не попадает в контекст сборки образа
├── Dockerfile                         # образ API-сервиса
├── pytest.ini                         # конфигурация тестов
├── compose.yaml                       # интеграционный стенд: api + client
├── requirements.txt                   # зафиксированные зависимости
├── client/                            # клиентский компонент стенда
│   ├── Dockerfile                     # отдельный образ клиента
│   ├── requirements.txt               # зависимости клиента (requests)
│   └── client.py                      # POST /predict и сохранение ответа
├── results/                           # сюда клиент кладет prediction.json
├── notebooks/
│   └── source_experiment.ipynb        # безопасная копия исходной заготовки
├── data_sample/
│   └── sample.csv                     # один тестовый объект Iris
├── models/
│   └── model.pkl                      # артефакт обучения (в Git не хранится)
├── app/
│   ├── __init__.py                    # пакет HTTP-слоя
│   ├── api.py                         # FastAPI-приложение: /health и /predict
│   └── schemas.py                     # Pydantic-схемы запроса и ответа
├── src/
│   ├── __init__.py                    # пакет прикладного слоя
│   ├── config.py                      # пути и параметры проекта
│   ├── train.py                       # обучение и сохранение модели
│   ├── model_service.py               # загрузка артефакта и предсказание
│   └── predict.py                     # CLI: загрузка модели и предсказание
└── tests/
    ├── conftest.py                    # общие фикстуры: модель, TestClient
    ├── test_model_service.py          # модульные тесты прикладного слоя
    ├── test_api.py                    # тесты HTTP-маршрутов и валидации
    └── test_predict.py                # smoke-тесты CLI-сценария
```

`models/model.pkl` намеренно исключен из репозитория: это воспроизводимый
артефакт, который создается командой `python src/train.py`.

## Установка

```bash
git clone https://gitverse.ru/<username>/<repository>.git
cd ml-integration-lab
python -m venv .venv
```

Активация окружения:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Обучение модели

```bash
python src/train.py
```

Скрипт загружает набор Iris, делит его на обучающую и отложенную выборки
(75/25, `random_state=42`), обучает логистическую регрессию, печатает accuracy
и сохраняет модель в `models/model.pkl`.

Ожидаемый вывод:

```text
accuracy=0.947
model_saved=...\ml-integration-lab\models\model.pkl
```

Значение accuracy может незначительно отличаться при других версиях библиотек.

Необязательный аргумент: `--model-path PATH` — сохранить модель в другой файл.

## Предсказание

```bash
python src/predict.py
```

Скрипт загружает сохраненную модель (**не обучая ее заново**), читает первую
строку `data_sample/sample.csv` и печатает предсказанный класс.

Ожидаемый вывод:

```text
predicted_class=0
predicted_name=setosa
```

Если модель еще не обучена, скрипт завершится с кодом 1 и напечатает в stderr
сообщение:

```text
error: файл модели не найден: ...\models\model.pkl. Сначала выполните: python src/train.py
```

Необязательные аргументы:

- `--model-path PATH` — путь к другому артефакту модели (например, сохраненному
  командой `python src/train.py --model-path PATH`);
- `--input PATH` — другой CSV-файл с одним объектом (формат описан ниже);
- `--format {text,json}` — формат вывода результата (по умолчанию `text`).

### Формат вывода

По умолчанию (`--format text`) результат печатается двумя строками
`ключ=значение` — так его удобно читать человеку. Для интеграции с другими
программами используйте `--format json`: весь вывод — один JSON-объект с теми
же полями.

```bash
python src/predict.py --format json
```

```json
{"predicted_class": 0, "predicted_name": "setosa"}
```

Ошибки (нет модели, нет входного файла, некорректные данные) всегда
печатаются в stderr, так что в stdout попадает только результат. В режиме
`json` ошибка тоже выводится JSON-объектом с ключом `error` (не-ASCII символы
экранируются как `\uXXXX`), а код возврата равен 1:

```json
{"error": "файл модели не найден: ..."}
```

Недопустимое значение (например, `--format xml`) отклоняется до загрузки
модели: скрипт печатает список допустимых форматов и завершается с кодом 2.

Формат входного CSV (заголовок обязателен, читается первая строка данных):

```csv
sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
```

## API-сервис

Тот же компонент доступен по HTTP. Приложение FastAPI объявлено в
[`app/api.py`](app/api.py), схемы запроса и ответа — в
[`app/schemas.py`](app/schemas.py), а загрузка модели и предсказание вынесены
в [`src/model_service.py`](src/model_service.py): HTTP-слой не знает, как
устроена модель, и не умеет её обучать.

Запуск сервиса (модель должна быть обучена заранее):

```bash
python src/train.py
python -m uvicorn app.api:app --reload
```

Сервис поднимается на `http://127.0.0.1:8000`. Модель загружается один раз при
старте приложения, а не на каждый запрос.

### GET /health

Состояние сервиса и готовность модели:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_ready":true}
```

Если артефакт модели не найден, сервис всё равно запускается, но возвращает
`"model_ready": false`, а `/predict` отвечает кодом 503.

### POST /predict

Принимает признаки одного объекта и возвращает класс:

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

```json
{"prediction":0,"predicted_name":"setosa"}
```

Валидацию входных данных выполняет Pydantic: все четыре поля обязательны и
должны быть числами больше нуля. При нарушении контракта FastAPI отвечает
кодом **422** и описывает, какое поле неверно, — приложение при этом не падает:

```json
{"detail":[{"type":"missing","loc":["body","petal_width"],"msg":"Field required"}]}
```

| Код ответа | Когда возникает |
| --- | --- |
| 200 | запрос корректен, предсказание выполнено |
| 422 | ошибка валидации Pydantic (нет поля, неверный тип, значение ≤ 0) |
| 503 | артефакт модели недоступен, нужно выполнить `python src/train.py` |

### Документация

FastAPI строит описание API по тем же Pydantic-схемам:

- `http://127.0.0.1:8000/docs` — Swagger UI;
- `http://127.0.0.1:8000/openapi.json` — машиночитаемая схема OpenAPI 3.1.

### Путь к модели

Артефакт берётся из переменной окружения `MODEL_PATH`, а если она не задана —
из `src/config.py` (`models/model.pkl`). Это нужно для запуска в контейнере:

```bash
MODEL_PATH=/app/models/model.pkl python -m uvicorn app.api:app
```

## Тестирование

```bash
pytest
```

Ожидаемый результат:

```text
31 passed
```

Проверки разделены по уровням — так понятно, на каком слое возникла ошибка:

| Уровень | Файл | Что проверяет |
| --- | --- | --- |
| Модульный (модель) | `tests/test_model_service.py` | `predict_one()` без HTTP: структура результата, узнаваемые объекты, отказ при неверном числе признаков, загрузка артефакта один раз |
| Модульный (CLI) | `tests/test_predict.py` | чтение CSV, форматы вывода `text`/`json`, ошибки CLI-сценария из практик 1–2 |
| API | `tests/test_api.py` | `GET /health` и успешный `POST /predict` через `TestClient` |
| Негативный | `tests/test_api.py` | некорректный JSON (нет поля, неверный тип, значение ≤ 0, пустое тело) → **422**; сервис остаётся живым |
| Интеграционный | `tests/test_model_service.py`, `tests/test_api.py` | сервис предсказывает тем же артефактом, что сохранён на диске; при отсутствии артефакта `/health` возвращает `model_ready=false`, а `/predict` — **503** |
| Контракт | `tests/test_api.py` | `/openapi.json` содержит оба маршрута и поля `PredictRequest`; `/docs` открывается |

`TestClient` поднимает приложение внутри процесса pytest, поэтому отдельный
`uvicorn` и браузер для тестов не нужны.

Тестовая конфигурация:

- `pytest.ini` — `pythonpath = . src`, чтобы тесты импортировали `app`, `src` и
  модули первой практики без установки проекта;
- `tests/conftest.py` — фикстуры: `trained_model_path` обучает модель один раз
  за сессию во временный каталог, `client` подменяет артефакт в сервисе и
  отдаёт `TestClient`, `client_without_model` моделирует отсутствие модели.

Тесты не зависят от наличия `models/model.pkl` в репозитории и не изменяют его.

Запуск отдельных уровней:

```bash
pytest tests/test_api.py -v
pytest tests/test_model_service.py -v
```

## Запуск в Docker

Требуется Docker (проверено на Docker Desktop 28.5.1). Образ описан в
[`Dockerfile`](Dockerfile), контекст сборки ограничен файлом
[`.dockerignore`](.dockerignore).

**Перед сборкой обучите модель:** артефакт `models/model.pkl` не хранится в Git,
а внутрь образа копируется готовым — обучение при сборке и в обработчике
запроса не выполняется.

```bash
python src/train.py
docker build -t ml-api:practice5 .
docker image ls ml-api
```

Запуск контейнера: порт 8000 контейнера публикуется как порт 8080 хоста, путь к
модели передаётся переменной окружения:

```bash
docker run -d --name ml-api-p5 -p 8080:8000 -e MODEL_PATH=/app/models/model.pkl ml-api:practice5
docker ps
```

Если порт 8080 на хосте уже занят другим приложением, подставьте любой
свободный порт — меняется только левая часть: `-p 8088:8000`.

Проверка API с хоста:

```bash
curl http://localhost:8080/health
# {"status":"ok","model_ready":true}

curl -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   --data-raw '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# {"prediction":0,"predicted_name":"setosa"}
```

Диагностика и остановка:

```bash
docker logs ml-api-p5           # журнал uvicorn и запросы
docker exec ml-api-p5 pwd       # /app — каталог из WORKDIR
docker exec ml-api-p5 ls -la /app
docker stats --no-stream ml-api-p5
docker stop ml-api-p5
docker rm ml-api-p5
```

Контейнер можно удалить и создать заново из того же образа — результат не
меняется, потому что состояние целиком описано образом и параметрами запуска.

Что важно в этом Dockerfile:

| Инструкция | Назначение |
| --- | --- |
| `FROM python:3.12-slim` | базовый образ с явным тегом версии |
| `WORKDIR /app` | рабочий каталог **внутри** образа; каталог на хосте не меняется |
| `COPY requirements.txt` → `RUN pip install` | зависимости ставятся до копирования кода, поэтому при правке кода слой с зависимостями берётся из кеша |
| `COPY app src models` | код приложения и готовый артефакт модели |
| `EXPOSE 8000` | документирует порт приложения, но **не** открывает порт на хосте — это делает `-p` |
| `CMD … --host 0.0.0.0` | внутри контейнера сервис должен слушать все интерфейсы: `127.0.0.1` был бы доступен только изнутри контейнера |

## Интеграционный стенд (Docker Compose)

Стенд поднимает два сервиса: `api` — тот же образ, что в практике 5, и `client` —
отдельный компонент, который делает один запрос `POST /predict` и сохраняет ответ
в `results/prediction.json`.

```text
Командная строка хоста
        │  http://localhost:8080/health
        ▼
┌────────────────┐      сеть app_net      ┌────────────────┐
│ api            │ ◄───────────────────── │ client         │
│ :8000          │  http://api:8000       │ POST /predict  │
└────────────────┘                        └───────┬────────┘
                                                  │ /results
                                                  ▼
                                        ./results на хосте
```

`api` — это DNS-имя сервиса внутри сети Compose. `localhost` внутри контейнера
`client` означает сам `client`, поэтому обращение к API идёт по `http://api:8000`.
Пользователь с хоста ходит на `http://localhost:8080`, потому что этот порт
опубликован параметром `ports`.

Запуск (модель должна быть обучена, см. раздел «Запуск в Docker»):

```bash
docker compose config         # проверка конфигурации
docker compose up --build -d
docker compose ps -a
```

`api` поднимается первым; `client` стартует только после того, как healthcheck
`api` перешёл в состояние `healthy` (`depends_on: condition: service_healthy`).
Клиент одноразовый: после успешного запроса он завершается со статусом
`Exited (0)` — это нормальное состояние, а не ошибка.

Проверка результата:

```bash
docker compose logs api
docker compose logs client
curl http://localhost:8080/health
cat results/prediction.json          # PowerShell: Get-Content .esults\prediction.json
```

```json
{
  "prediction": 0,
  "predicted_name": "setosa"
}
```

Если порт 8080 на хосте занят, задайте другой через переменную `API_HOST_PORT`
(в `compose.yaml` она подставляется в левую часть `ports`):

```bash
API_HOST_PORT=8088 docker compose up --build -d
```

### Диагностический опыт: почему не `localhost`

```bash
docker compose run --rm -e API_URL=http://localhost:8000 client
```

```text
error: запрос к API не выполнен: ... Failed to establish a new connection:
[Errno 111] Connection refused
```

Внутри контейнера `client` адрес `localhost` указывает на сам контейнер клиента,
где на порту 8000 никто не слушает. Опубликованный порт хоста здесь тоже не
помогает: контейнеры общаются напрямую по сети Compose. Правильный адрес —
`http://api:8000`, он задан в `compose.yaml`:

```bash
docker compose run --rm client     # HTTP 200, ответ сохранён
```

### Остановка

```bash
docker compose down
cat results/prediction.json
```

`down` удаляет контейнеры и сеть проекта, но `results/prediction.json` остаётся
на хосте: каталог подключён как bind mount `./results:/results`, где левая часть —
путь на хосте, правая — путь внутри контейнера.

Сам файл результата в Git не хранится (он в `.gitignore`), каталог остаётся в
репозитории за счёт `results/.gitkeep`.

## Конфигурация и зависимости

Параметры проекта собраны в [`src/config.py`](src/config.py):

| Параметр | Значение | Назначение |
| --- | --- | --- |
| `PROJECT_ROOT` | каталог репозитория | база для всех путей |
| `MODEL_PATH` | `models/model.pkl` | артефакт модели (переопределяется переменной окружения `MODEL_PATH`) |
| `SAMPLE_PATH` | `data_sample/sample.csv` | входные данные по умолчанию |
| `FEATURE_COLUMNS` | 4 столбца Iris | интерфейс входных данных |
| `OUTPUT_FORMATS` | `("text", "json")` | допустимые форматы вывода `predict.py` |
| `DEFAULT_OUTPUT_FORMAT` | `"text"` | формат вывода по умолчанию |
| `TEST_SIZE` | `0.25` | доля отложенной выборки |
| `RANDOM_STATE` | `42` | воспроизводимость разбиения и обучения |
| `MAX_ITER` | `300` | предел итераций логистической регрессии |

Пути к модели и входному файлу, а также формат вывода можно переопределить
аргументами командной строки, не меняя код.

Зависимости зафиксированы по версиям в `requirements.txt`:
`scikit-learn`, `joblib`, `pandas` — модель; `fastapi`, `uvicorn`, `pydantic` —
API-сервис; `pytest`, `httpx` — тесты. Точные версии указаны в
`requirements.txt`.
Проект проверен на Python 3.13.

## Публикация в GitVerse

```bash
git branch -M main
git remote add origin https://gitverse.ru/<username>/<repository>.git
git push -u origin main
```

## Внесение изменений

Изменения попадают в `main` не прямой правкой, а через отдельную ветку и
merge request (pull request):

```bash
git switch main
git pull
git switch -c feature/<краткое-имя>
# изменить код, тесты и README; перед каждым коммитом — git status и git diff
pytest
git push -u origin feature/<краткое-имя>
```

В описании merge request указываются цель изменения, способ проверки и риски.
Перед слиянием `main` подтягивается в ветку (`git merge origin/main`),
конфликты разрешаются вручную, после чего тесты запускаются повторно.

## Ограничения

- Проект учебный: цель работы — инженерная организация репозитория, а не
  качество модели.
- Датасет Iris маленький и встроен в scikit-learn; предобработка отсутствует.
- Модель (логистическая регрессия с параметрами по умолчанию) не подбиралась и
  не валидировалась кросс-валидацией; accuracy приведена справочно.
- `predict.py` обрабатывает один объект — первую строку CSV; пакетное
  предсказание и сервисный интерфейс (API) не реализованы.
- Артефакт модели не версионируется и не хранится в Git: после клонирования
  репозитория нужно выполнить `python src/train.py`.
