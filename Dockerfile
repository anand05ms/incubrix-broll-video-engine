FROM python:3.12-slim AS tests
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng fonts-dejavu-core fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /service
COPY pyproject.toml README.md ./
COPY app ./app
COPY tests ./tests
RUN pip install ".[test]"
CMD ["python", "-m", "pytest"]

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PIP_NO_CACHE_DIR=1

RUN apt-get update \
&& apt-get install -y --no-install-recommends ffmpeg espeak-ng fonts-dejavu-core fonts-noto-core \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install ".[asr,vision,kokoro]"

RUN useradd --create-home --uid 10001 appuser \
&& mkdir -p /service/data \
&& chown -R appuser:appuser /service
ENV TORCH_HOME=/service/data/model-cache
ENV HF_HOME=/service/data/model-cache/huggingface
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
