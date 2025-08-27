# ========= stage: builder =========
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED="1" \
    PIP_NO_CACHE_DIR="1"

# Build arg: 0=wheel-only (small; build as linux/amd64), 1=portable (adds GDAL/PROJ & toolchain)
ARG PORTABLE_BUILD=0

# Base OS deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv + modern pip
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir uv \
 && uv --version

# ---- Export ONLY third-party runtime deps (no hashes). We filter out any editable/self lines.
# NOTE: We export BEFORE copying src/ to avoid uv deciding this is a local edit; still filter to be safe.
RUN set -euo pipefail; \
    uv export --no-group dev --no-group test --format=requirements-txt --no-hashes -o /tmp/req.raw.txt; \
    awk ' \
      BEGIN{IGNORECASE=1} \
      # drop editable flags
      /^-e[[:space:]]/ || /^--editable[[:space:]]/ {next} \
      # drop local/self refs
      /@ file:/ || /file:\/\// {next} \
      # drop our own package if present
      /^eopf-geozarr([[:space:]]|==|$)/ {next} \
      # drop comments/blank lines
      /^[[:space:]]*#/ || /^[[:space:]]*$/ {next} \
      {print} \
    ' /tmp/req.raw.txt > /tmp/requirements.txt; \
    echo "----- filtered requirements (head) -----"; \
    sed -n '1,80p' /tmp/requirements.txt

# ---- If PORTABLE, add toolchain + GDAL/PROJ for sdist builds (arm64 etc.)
RUN if [ "$PORTABLE_BUILD" = "1" ]; then \
      apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential gdal-bin libgdal-dev proj-bin libproj-dev \
      && rm -rf /var/lib/apt/lists/* ; \
    fi

# ---- Install third-party deps, then your package (no re-resolve of deps)
RUN if [ "$PORTABLE_BUILD" = "1" ]; then \
      pip install --no-cache-dir -r /tmp/requirements.txt ; \
    else \
      PIP_ONLY_BINARY=":all:" pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt ; \
    fi

# Now copy source and install the project itself without deps (pure python install)
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Optional: byte-compile
RUN python -m compileall -q /usr/local/lib/python3.11/site-packages || true

# ========= stage: runtime =========
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED="1"

# Tiny libs manylinux wheels often need
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local /usr/local

# Argo Script template supplies the command
CMD ["python", "-V"]
