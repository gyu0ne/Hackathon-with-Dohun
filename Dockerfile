FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    OMP_NUM_THREADS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-kor \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY research/scripts/install_tessdata.py /tmp/install_tessdata.py
RUN python /tmp/install_tessdata.py \
    && rm /tmp/install_tessdata.py

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY research/scripts/warm_paddleocr.py /tmp/warm_paddleocr.py
RUN python /tmp/warm_paddleocr.py \
    && rm /tmp/warm_paddleocr.py

COPY app ./app
COPY research ./research
COPY ocr_training ./ocr_training
COPY tests ./tests
COPY pyproject.toml ./

RUN mkdir -p /app/data /app/research/manifests

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
