FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .

RUN uv pip install --system .

EXPOSE 8765

# The endpoint path is the connector's only secret, so it comes from the
# MCP_PATH environment variable (set it in Render) and is never committed.
# `:?` makes the container refuse to start if it's missing, rather than fall
# back to a guessable public path.
CMD garmin-mcp --transport streamable-http --host 0.0.0.0 --port $PORT --path "/${MCP_PATH:?Set MCP_PATH to a long random string}/mcp"
