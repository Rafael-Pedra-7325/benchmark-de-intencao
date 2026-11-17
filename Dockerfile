FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY benchmark_intencao ./benchmark_intencao
COPY data ./data
COPY resultados ./resultados
COPY app.py ./

RUN pip install --no-cache-dir -e ".[llm,web]"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
