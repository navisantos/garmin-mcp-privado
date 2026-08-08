FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .

RUN uv pip install --system .

EXPOSE 8765

CMD garmin-mcp --transport streamable-http --host 0.0.0.0 --port $PORT --path /5Y6YsuQJjXHpokFt_jHslvh4QkLc2iBEFRwrKuLlUNU/mcp