FROM python:3.12-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cascade_pipeline.py .
COPY serving/ ./serving/

# Non-root. /data is created and chowned here so the named volume inherits this
# ownership on first mount — otherwise the volume lands root-owned and unwritable.
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app /app /data
USER app

# Flat imports (`import config`) — run from inside serving/.
WORKDIR /app/serving

ENV PYTHONUNBUFFERED=1 \
    REDIS_URL=redis://redis:6379/0 \
    RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/ \
    STORAGE_DIR=/data/bucket
# DB_URL comes from compose (it carries the Postgres password).

# ponytail: no entrypoint script — compose sets `command: python rabbitmq.py` for the worker.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
