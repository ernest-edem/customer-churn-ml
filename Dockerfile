FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY src ./src
COPY config ./config

RUN mkdir -p models \
    && python -c "import hashlib; import urllib.request; from pathlib import Path; url='https://github.com/ernest-edem/customer-churn-ml/releases/download/v1.0.0/model.joblib'; path=Path('models/model.joblib'); urllib.request.urlretrieve(url, path); expected='26a5540358d24c20920c560056d955acd054660dfa98299fc4250c161cbec54f'; actual=hashlib.sha256(path.read_bytes()).hexdigest(); print(f'Model SHA-256: {actual}'); assert actual == expected, f'Unexpected model SHA-256: {actual}'"

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]