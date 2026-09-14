# Pinned by digest, not tag, so this build is the same image tomorrow as
# it is today. Verified against the real digest at build time:
#   docker pull python:3.13-slim
#   docker inspect python:3.13-slim --format='{{index .RepoDigests 0}}'
FROM python@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

# uv, pinned the same way this book pins everything else: an exact
# version, copied straight from its own published image rather than a
# curl-pipe-to-shell install script.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so an application code change does not invalidate
# this layer. --frozen refuses to build if uv.lock is out of date instead
# of silently re-resolving, the container build fails the same way a
# reader's local `uv sync --locked` would.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY config/ config/
RUN uv sync --frozen --no-dev

EXPOSE 8000

# --no-sync matters here: without it, `uv run` re-checks pyproject.toml
# and uv.lock against the environment on every container start and
# resyncs if anything looks stale, including reaching out for dev
# dependencies this image was built with --no-dev specifically to leave
# out. That defeats the point of a pinned build, and fails outright in
# any deployment without egress. --no-sync runs with exactly what was
# installed at build time, nothing more.
CMD ["uv", "run", "--no-sync", "uvicorn", "reliable_agents_labs.api:app", "--host", "0.0.0.0", "--port", "8000"]
