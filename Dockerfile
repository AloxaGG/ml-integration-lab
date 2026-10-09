# Образ API-сервиса. Модель не обучается при сборке: в образ копируется
# готовый артефакт models/model.pkl, созданный командой python src/train.py.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.pkl

# Рабочий каталог внутри образа и будущего контейнера
WORKDIR /app

# Зависимости копируются и ставятся до исходного кода: при правке кода
# повторная сборка берет этот слой из кеша и не переустанавливает пакеты.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения, клиентская страница и артефакт модели
COPY app ./app
COPY src ./src
COPY static ./static
COPY models ./models

# Документирует порт приложения; публикацию на хост задает -p при docker run
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
