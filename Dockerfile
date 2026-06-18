FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FEDSEG_MODEL_PATH=/app/model/fedseg_model.pt
ENV FEDSEG_API_URL=http://127.0.0.1:8000
ENV PYTHONUNBUFFERED=1

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
