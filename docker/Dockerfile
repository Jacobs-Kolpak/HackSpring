FROM python:3.11-slim

WORKDIR /app

# Системные зависимости:
#   libpq-dev, gcc — для psycopg2
#   libmupdf-dev, mupdf-tools — для PyMuPDF (fitz)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libpq-dev gcc g++ \
       ffmpeg fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Директория для данных (vector store, uploads, audio)
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
