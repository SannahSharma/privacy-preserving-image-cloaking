FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Render sets $PORT; default to 8000 for local runs
ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}