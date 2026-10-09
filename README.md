# ML Integration Lab

[![CI](https://github.com/AloxaGG/ml-integration-lab/actions/workflows/ci.yaml/badge.svg)](https://github.com/AloxaGG/ml-integration-lab/actions/workflows/ci.yaml)
[![Publish API documentation](https://github.com/AloxaGG/ml-integration-lab/actions/workflows/publish-api-docs.yaml/badge.svg)](https://github.com/AloxaGG/ml-integration-lab/actions/workflows/publish-api-docs.yaml)

Учебный пример по дисциплине «Разработка и интеграция» (магистратура 01.04.02):
путь от исследовательского notebook до ML-сервиса с тестами, контейнером,
интеграционным стендом, CI/CD и опубликованной документацией.

Репозиторий показывает не только итоговый код, но и **процесс**: каждая
практическая работа выполнена в отдельной ветке и влита через pull request с
описанием цели, способа проверки и рисков.

- **Документация API:** <https://aloxagg.github.io/ml-integration-lab/api/>
- **Запуски CI/CD:** [вкладка Actions](https://github.com/AloxaGG/ml-integration-lab/actions)
- **История изменений:** [список pull request](https://github.com/AloxaGG/ml-integration-lab/pulls?q=is%3Apr+is%3Amerged)

## Содержание по практическим работам

| Практика | Что добавлено | Разделы README | Pull request |
| --- | --- | --- | --- |
| 1 | структура проекта, разделение обучения и инференса, пути от корня, smoke-тесты | [Структура](#структура-проекта), [Обучение](#обучение-модели), [Предсказание](#предсказание) | — |
| 2 | параметр `--format text\|json`, работа через ветку, review и разрешение конфликта | [Формат вывода](#формат-вывода), [Внесение изменений](#внесение-изменений) | [#1](https://github.com/AloxaGG/ml-integration-lab/pull/1) |
| 3 | FastAPI-сервис: `/health`, `/predict`, Pydantic-схемы, OpenAPI | [API-сервис](#api-сервис) | [#2](https://github.com/AloxaGG/ml-integration-lab/pull/2) |
| 4 | тесты модульные, API, негативные и интеграционные | [Тестирование](#тестирование) | [#3](https://github.com/AloxaGG/ml-integration-lab/pull/3) |
| 5 | `Dockerfile`, `.dockerignore`, запуск в контейнере | [Запуск в Docker](#запуск-в-docker) | [#4](https://github.com/AloxaGG/ml-integration-lab/pull/4) |
| 6 | `compose.yaml`: API + клиент, сеть, healthcheck, bind mount | [Интеграционный стенд](#интеграционный-стенд-docker-compose) | [#5](https://github.com/AloxaGG/ml-integration-lab/pull/5) |
| 7 | CI: проверки, тесты и сборка образа | [CI/CD](#cicd) | [#6](https://github.com/AloxaGG/ml-integration-lab/pull/6) |
| 8 | workflow доставки в режиме dry run и откат | [Доставка (dry run)](#доставка-dry-run) | [#7](https://github.com/AloxaGG/ml-integration-lab/pull/7) |
| 9 | автоматическая публикация Swagger UI | [Документация API](#документация-api-опубликованная) | [#8](https://github.com/AloxaGG/ml-integration-lab/pull/8) |
| Предзачётная | профиль признаков `GET /features`, ранняя валидация, страница на `GET /` | [Профиль признаков](#профиль-признаков-и-ранняя-валидация), [Клиентская страница](#клиентская-страница) | [#11](https://github.com/AloxaGG/ml-integration-lab/pull/11) |

## Быстрый старт

```bash
git clone https://github.com/AloxaGG/ml-integration-lab.git
cd ml-integration-lab
python -m venv .venv && .venv/Scripts/Activate.ps1   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # 31 passed
python src/train.py                      # обучение модели -> models/model.pkl
python -m uvicorn app.api:app --reload   # сервис на http://127.0.0.1:8000
```

Подробности по каждому шагу — в разделах ниже.

## Цель проекта

Преобразовать исследовательский ML-код (notebook) в минимальный инженерный
проект, пригодный для дальнейшей интеграции в программную систему.

Прикладная задача модели — классификация ирисов (набор Iris, 4 числовых
признака, 3 класса) с помощью логистической регрессии. Проект разделяет два
независимых сценария:

- **обучение** — `src/train.py` обучает модель и сохраняет артефакт на диск;
- **инференс** — `src/predict.py` загружает *уже сохраненный* артефакт и
  предсказывает класс для одного объекта из CSV.

## Исходная заготовка (практика 1)

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
├── docs/                              # исходники сайта документации
│   ├── index.html                     # стартовая страница
│   └── api/                           # Swagger UI и его локальные файлы
├── scripts/
│   └── export_openapi.py              # экспорт схемы из app.openapi()
├── requirements.txt                   # зафиксированные зависимости
├── client/                            # клиентский компонент стенда
│   ├── Dockerfile                     # отдельный образ клиента
│   ├── requirements.txt               # зависимости клиента (requests)
│   └── client.py                      # POST /predict и сохранение ответа
├── results/                           # сюда клиент кладет prediction.json
├── static/
│   └── index.html                     # клиентская страница, отдается по GET /
├── notebooks/
│   ├── source_experiment.ipynb        # безопасная копия исходной заготовки
│   ├── source_experiment_updated.ipynb # эксперимент: построение профиля признаков
│   └── new_functionality.ipynb        # эксперимент: проверка объекта по профилю
├── data_sample/
│   └── sample.csv                     # один тестовый объект Iris
├── models/
│   └── model.pkl                      # артефакт обучения (в Git не хранится)
├── app/
│   ├── __init__.py                    # пакет HTTP-слоя
│   ├── api.py                         # FastAPI-приложение: /, /health, /features, /predict
│   └── schemas.py                     # Pydantic-схемы запроса и ответа
├── src/
│   ├── __init__.py                    # пакет прикладного слоя
│   ├── config.py                      # пути и параметры проекта
│   ├── train.py                       # обучение и сохранение модели
│   ├── model_service.py               # загрузка артефакта и предсказание
│   ├── feature_profile.py             # профиль признаков и проверка границ
│   └── predict.py                     # CLI: загрузка модели и предсказание
└── tests/
    ├── conftest.py                    # общие фикстуры: модель, TestClient
    ├── test_feature_profile.py        # модульные тесты профиля и валидации
    ├── test_model_service.py          # модульные тесты прикладного слоя
    ├── test_api.py                    # тесты HTTP-маршрутов и валидации
    └── test_predict.py                # smoke-тесты CLI-сценария
```

`models/model.pkl` намеренно исключен из репозитория: это воспроизводимый
артефакт, который создается командой `python src/train.py`.

## Установка

```bash
git clone https://github.com/AloxaGG/ml-integration-lab.git
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
{"class_id":0,"class_name":"setosa"}
```

Запрос проходит две проверки. Сначала Pydantic проверяет схему: все четыре
поля обязательны и должны быть числами больше нуля. Затем значения сверяются с
профилем признаков — подробности в разделе
[«Профиль признаков и ранняя валидация»](#профиль-признаков-и-ранняя-валидация).
Обе проверки отвечают кодом **422** в одинаковом формате, приложение при этом
не падает:

```json
{
  "detail": "Некорректный запрос: petal_width: Field required",
  "errors": [{"type": "missing", "loc": ["body", "petal_width"], "msg": "Field required"}]
}
```

| Код ответа | Когда возникает |
| --- | --- |
| 200 | запрос корректен, предсказание выполнено |
| 422 | ошибка схемы (нет поля, неверный тип, значение ≤ 0) или значение вне профиля признаков |
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

## Профиль признаков и ранняя валидация

Сервис сообщает пользователю допустимые диапазоны признаков и отклоняет заведомо
невозможные значения **до** обращения к модели. Контракт становится прозрачным:
клиент заранее знает границы, а модель не получает данные, по которым не училась.

Профиль — это минимум, максимум и среднее по каждому признаку обучающего набора
Iris. Он вычисляется один раз при старте приложения и дальше берётся из кеша
([`src/feature_profile.py`](src/feature_profile.py)). Код перенесён из
экспериментов `notebooks/source_experiment_updated.ipynb` (построение профиля) и
`notebooks/new_functionality.ipynb` (проверка объекта) — ноутбуки сохранены как
воспроизводимые эксперименты, бизнес-логика живёт в модулях проекта.

### Маршруты

| Метод и путь | Назначение | Успешный ответ |
| --- | --- | --- |
| `GET /` | клиентская страница из `static/index.html` | HTML |
| `GET /health` | готовность сервиса и модели | `status`, `model_ready` |
| `GET /features` | профиль четырёх признаков | `{признак: {min, max, mean}}` |
| `POST /predict` | проверка по профилю и прогноз | `class_id`, `class_name` |

### GET /features

```bash
curl http://127.0.0.1:8000/features
```

```json
{
  "sepal_length": {"min": 4.3, "max": 7.9, "mean": 5.843333333333334},
  "sepal_width":  {"min": 2.0, "max": 4.4, "mean": 3.0573333333333337},
  "petal_length": {"min": 1.0, "max": 6.9, "mean": 3.7580000000000005},
  "petal_width":  {"min": 0.1, "max": 2.5, "mean": 1.1993333333333336}
}
```

Формат пригоден фронтенду без преобразования: ключ — имя признака, значение —
объект с границами и средним. Маршрут не обращается к модели и работает, даже
если артефакт ещё не обучен.

### Проверка диапазонов в POST /predict

Границы **включительны**: значение, равное `min` или `max`, допустимо. Если хотя
бы один признак выходит за границы, сервис возвращает **422** и не вызывает
модель. В ответе — читаемое сообщение и список всех нарушений сразу:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":4.0}'
```

```json
{
  "detail": "Значения вне допустимого диапазона: petal_width=4 вне диапазона [0.1; 2.5]",
  "errors": [{"feature": "petal_width", "value": 4.0, "min": 0.1, "max": 2.5}]
}
```

Значение не «подгоняется» к ближайшей границе: молчаливое исправление скрыло бы
ошибку клиента и дало бы прогноз, которого пользователь не запрашивал.

Ошибки схемы (нет поля, неверный тип) отдаются в том же формате, поэтому клиенту
достаточно одного способа разбора:

```json
{
  "detail": "Некорректный запрос: petal_width: Field required",
  "errors": [{"type": "missing", "loc": ["body", "petal_width"], "msg": "Field required"}]
}
```

## Клиентская страница

Страница лежит в [`static/index.html`](static/index.html) и отдаётся тем же
приложением по `GET /`:

```bash
python src/train.py
python -m uvicorn app.api:app --reload
# открыть http://127.0.0.1:8000/
```

Как она работает:

1. при загрузке запрашивает `/features` и `/health`;
2. по профилю заполняет у числовых полей атрибуты `min`, `max` и подсказку со
   средним значением, индикатор вверху показывает состояние сервиса;
3. перед отправкой браузер проверяет значения сам, затем отправляет Ajax-запрос
   на `/predict`;
4. при 422 текст из поля `detail` показывается рядом с формой, при успехе класс
   выводится в отдельной карточке.

Все запросы страницы относительные (`/features`, `/predict`), поэтому она
работает с того же origin, что и API, и **CORS не нужен**.

### Клиентская проверка и серверная — зачем обе

| Проверка | Для чего | Можно ли ей доверять |
| --- | --- | --- |
| В браузере (`min`/`max` у поля, проверка перед отправкой) | удобство: ошибка видна сразу, без сетевого запроса | нет: её можно отключить, обойти через `curl` или изменить код страницы |
| На сервере (`validate_feature_ranges` до вызова модели) | защита контракта и модели от недопустимых данных | да: через неё проходит любой клиент |

Серверная проверка обязательна, клиентская — только дополнение к ней. Браузер
улучшает интерфейс, но решение о допустимости данных принимает сервис.

## Тестирование

```bash
pytest
```

Ожидаемый результат:

```text
45 passed
```

Проверки разделены по уровням — так понятно, на каком слое возникла ошибка:

| Уровень | Файл | Что проверяет |
| --- | --- | --- |
| Модульный (профиль) | `tests/test_feature_profile.py` | построение профиля, включительные границы, список всех нарушений, неизменяемость кеша |
| Модульный (модель) | `tests/test_model_service.py` | `predict_one()` без HTTP: структура результата, узнаваемые объекты, отказ при неверном числе признаков, загрузка артефакта один раз |
| Модульный (CLI) | `tests/test_predict.py` | чтение CSV, форматы вывода `text`/`json`, ошибки CLI-сценария из практик 1–2 |
| API | `tests/test_api.py` | `GET /health` и успешный `POST /predict` через `TestClient` |
| Негативный | `tests/test_api.py` | некорректный JSON (нет поля, неверный тип, значение ≤ 0, пустое тело) → **422**; сервис остаётся живым |
| Интеграционный | `tests/test_model_service.py`, `tests/test_api.py` | сервис предсказывает тем же артефактом, что сохранён на диске; при отсутствии артефакта `/health` возвращает `model_ready=false`, а `/predict` — **503** |
| Профиль и страница | `tests/test_api.py` | `GET /features` отдаёт профиль всех признаков; объект на границе принимается; значение вне диапазона → 422 с именем признака, значением и границами; отклонённый запрос не доходит до модели; `GET /` отдаёт HTML |
| Контракт | `tests/test_api.py` | `/openapi.json` содержит маршруты и поля `PredictRequest`; `/docs` открывается |

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
# {"class_id":0,"class_name":"setosa"}
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
cat results/prediction.json          # PowerShell: Get-Content .\results\prediction.json
esults\prediction.json
```

```json
{
  "class_id": 0,
  "class_name": "setosa"
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

## CI/CD

Конвейер описан в [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) и
хранится в репозитории вместе с кодом: он меняется через те же ветки и pull
request, поэтому его история видна наравне с историей приложения.

**События запуска**

| Событие | Когда срабатывает | Зачем |
| --- | --- | --- |
| `push` в `main` | изменения в `app/`, `src/`, `client/`, `tests/`, `Dockerfile`, `requirements.txt`, `pytest.ini`, сам workflow | основная ветка всегда проверена |
| `pull_request` | любой PR | ошибки видны до слияния |
| `workflow_dispatch` | ручной запуск из интерфейса | перепроверка без новых коммитов |

**Jobs**

1. `check` — установка зависимостей, проверка синтаксиса `python -m compileall`
   и запуск `pytest -q`. Тесты идут без ручного старта uvicorn и без браузера.
2. `build-image` — связана с `check` через `needs`, поэтому стартует только
   после успешных проверок. Обучает артефакт модели (`python src/train.py`),
   собирает образ по `Dockerfile` и прогоняет smoke-тест: запускает контейнер и
   проверяет `/health` и `/predict`. Образ никуда не публикуется (`push: false`).

Сборка выполняется на runner-е средствами buildx: образ собирается в
изолированной среде CI, а не на компьютере студента, и не зависит от локально
установленного Docker.

Артефакт `models/model.pkl` не хранится в Git, поэтому в CI он создаётся до
сборки образа. Внутри `Dockerfile` обучения нет — образ только копирует готовый
файл.

**Локальная проверка перед push** (те же команды, что и в CI):

```bash
pytest -q
python -m compileall -q app src client
docker build -t ml-api:local .
```

Ссылка на запуски: [вкладка Actions репозитория](https://github.com/AloxaGG/ml-integration-lab/actions).

### Доставка (dry run)

Отдельный workflow
[`.github/workflows/delivery.yaml`](.github/workflows/delivery.yaml) показывает
CD-сценарий: какая версия доставляется, какие проверки должны пройти до
обновления сервиса, какими командами сервис обновляется, как выполняется smoke
test и как откатиться на предыдущую версию.

**Тестового сервера в учебной работе нет**, поэтому реальная доставка не
выполняется: jobs печатают в лог команды, которые выполнили бы обновление.
Адреса `registry.example.local` и `test.example.local` — заглушки.

Запуск только ручной: вкладка **Actions** → **Test Delivery Dry Run** → **Run
workflow**. Доставку в тестовую среду выкатывают осознанно, а не на каждый push.

| Job | Зависимости | Что делает |
| --- | --- | --- |
| `build` | — | формирует тег версии `test-<первые 12 символов SHA>` и публикует его как output `image-tag` через `$GITHUB_OUTPUT` |
| `smoke_api_tests_stub` | `build` | печатает команды `curl` для `/health` и `/predict` проверяемого образа |
| `docs_checks` | `build` | проверяет, что README на месте и содержит описание доставки |
| `deploy_dry_run` | `build`, `smoke_api_tests_stub`, `docs_checks` | печатает команды `docker pull` / `stop` / `rm` / `run`, smoke test и команды отката |

Проверки `smoke_api_tests_stub` и `docs_checks` зависят только от `build`,
поэтому идут параллельно. `deploy_dry_run` ждёт их все через `needs` и
запускается при условии `success() && github.ref == 'refs/heads/main'`: доставка
не должна начинаться, если хоть одна проверка упала или запуск сделан не из
основной ветки. Тег берётся из `needs.build.outputs.image-tag` — он задаётся в
одном месте и не может разойтись между jobs.

Команды, которые печатает `deploy_dry_run`:

```bash
docker pull registry.example.local/ml-api:test-<sha>
docker stop ml-api-test || true
docker rm ml-api-test || true
docker run -d --name ml-api-test -p 8080:8000   -e MODEL_PATH=/app/models/model.pkl registry.example.local/ml-api:test-<sha>
curl -f http://test.example.local:8080/health
```

**Откат** выполняется теми же командами, но с предыдущим проверенным тегом
(`PREVIOUS_TAG=previous-stable`). В рабочем проекте такой тег хранится в
registry (например, метка `stable`, которая переставляется только после
успешного smoke test), чтобы всегда была версия, к которой можно вернуться.

**Как заменить dry run на реальную доставку.** Когда появится тестовый сервер,
`deploy_dry_run` заменяется на `deploy_test`, а команды `echo` — на `ssh` или
вызовы API сервера. Значения `DEPLOY_HOST`, `DEPLOY_USER`, `SSH_KEY`,
`REGISTRY_TOKEN` должны храниться в секретах CI, а не в репозитории. В этой
работе секреты не нужны и не добавляются.

## Документация API (опубликованная)

Swagger UI опубликован автоматически:
**https://aloxagg.github.io/ml-integration-lab/api/**
(стартовая страница — https://aloxagg.github.io/ml-integration-lab/,
схема — https://aloxagg.github.io/ml-integration-lab/api/openapi.json).

Как это работает:

1. workflow [`.github/workflows/publish-api-docs.yaml`](.github/workflows/publish-api-docs.yaml)
   запускается при push в `main` (и вручную через **Run workflow**);
2. job `build` импортирует приложение и вызывает `app.openapi()` через
   [`scripts/export_openapi.py`](scripts/export_openapi.py), сохраняя схему в
   `site/api/openapi.json`, проверяет, что JSON валиден и файлы Swagger UI на
   месте, и отдаёт каталог `site/` как артефакт Pages;
3. job `deploy` публикует этот артефакт.

Схема **не хранится в репозитории**: она собирается в CI из кода того коммита,
который запустил публикацию, поэтому опубликованная документация не может
разойтись с кодом. Веб-сервер при этом не запускается — `app.openapi()`
возвращает схему прямо из объекта приложения.

Файлы Swagger UI (`swagger-ui.css`, `swagger-ui-bundle.js`) лежат в
`docs/api/` — CDN не используется, сайт работает на файлах репозитория. Все
пути в `docs/api/index.html` относительные (`./openapi.json`), потому что сайт
публикуется в подкаталоге `/<имя-репозитория>/`.

| Путь в репозитории | Назначение |
| --- | --- |
| `docs/index.html` | стартовая страница со ссылкой на `./api/` |
| `docs/api/index.html` | страница Swagger UI, `url: './openapi.json'` |
| `docs/api/swagger-ui.css`, `docs/api/swagger-ui-bundle.js` | локальные файлы Swagger UI |
| `scripts/export_openapi.py` | импортирует `app` и сохраняет `app.openapi()` |
| `.github/workflows/publish-api-docs.yaml` | собирает `site/` и публикует его |

Локальная проверка экспорта схемы:

```bash
python scripts/export_openapi.py --output site/api/openapi.json
```

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

## Публикация репозитория

Если проект собирается с нуля, первая отправка в удалённый репозиторий выглядит так:

```bash
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
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
  предсказание не реализовано (для одного объекта есть API и страница).
- Профиль признаков построен по исходному набору Iris: он описывает границы
  обучающих данных, а не физически возможные значения.
- Артефакт модели не версионируется и не хранится в Git: после клонирования
  репозитория нужно выполнить `python src/train.py`.
