# Отдельный образ клиента: у него свой контекст сборки, свои зависимости
# (requests вместо scikit-learn/FastAPI) и своя команда запуска.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /client

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY client.py ./

CMD ["python", "client.py"]
