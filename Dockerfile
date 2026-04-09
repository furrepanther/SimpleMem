# SimpleMem MCP Server — multi-stage, layer-cached build
FROM python:3.12-slim AS base

LABEL maintainer="SimpleMem"
LABEL description="SimpleMem MCP Server with LanceDB vector storage"
LABEL org.opencontainers.image.source="https://github.com/user/SimpleMem"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

COPY MCP/requirements.txt /app/MCP/requirements.txt
RUN pip install --no-cache-dir -r /app/MCP/requirements.txt

COPY . .
RUN chmod +x /app/scripts/docker-entrypoint.sh

WORKDIR /app/MCP

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app

ENV DATA_DIR=/app/MCP/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
