FROM python:3.13-slim

# ffmpeg habilita el alimentador de audio desde archivos y streams.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY poliglota ./poliglota
RUN pip install --no-cache-dir -e .
COPY web ./web
COPY scripts ./scripts

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["poliglota"]
