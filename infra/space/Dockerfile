# Hugging Face Space (Docker SDK): the CPU commentary API.
# Serves a quantized GGUF of the fine-tune via llama.cpp behind the FastAPI app.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama-cpp-python from the prebuilt CPU wheel index (no compile), plus the serving deps.
RUN pip install --no-cache-dir \
        llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    && pip install --no-cache-dir \
        "fastapi>=0.115,<1" "uvicorn>=0.30,<1" \
        "huggingface-hub>=0.25,<1" "pydantic>=2.6,<3" "pydantic-settings>=2.6,<3" \
        "structlog>=24.4,<25" "jinja2>=3.1,<4"

# Pull the library code (app/ + configs/) and the Space entry from the repo.
ARG REPO=https://github.com/kartikeyamandhar/cover-drive
RUN git clone --depth 1 "$REPO" /src \
    && cp -r /src/app /src/configs /src/infra/space/server.py ./ \
    && rm -rf /src

# Caches and the model live under /tmp (the writable dir on Spaces).
ENV HF_HOME=/tmp/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
