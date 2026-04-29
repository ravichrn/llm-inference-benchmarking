FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-cache

COPY src ./src

EXPOSE 8010
CMD ["uv", "run", "uvicorn", "llm_inference_benchmarking.service_app:app", "--host", "0.0.0.0", "--port", "8010"]
