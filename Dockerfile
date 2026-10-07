# ai-trader: bot (default) and dashboard share this image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tini sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 trader
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
    && mkdir -p /app/data \
    && chown trader:trader /app/data

USER trader

# Claude Code CLI (native installer) for `provider: claude_code`. Pin a version with
# --build-arg CLAUDE_CODE_VERSION=x.y.z; set INSTALL_CLAUDE_CODE=0 to skip it.
ARG INSTALL_CLAUDE_CODE=1
ARG CLAUDE_CODE_VERSION=stable
RUN if [ "$INSTALL_CLAUDE_CODE" = "1" ]; then \
        curl -fsSL https://claude.ai/install.sh | bash -s "$CLAUDE_CODE_VERSION"; \
    fi
ENV PATH="/home/trader/.local/bin:${PATH}" \
    DISABLE_AUTOUPDATER=1

COPY --chown=trader config ./config

ENTRYPOINT ["tini", "--"]
CMD ["ai-trader", "run"]
