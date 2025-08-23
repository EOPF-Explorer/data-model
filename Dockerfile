FROM python:3.11-slim

# system deps for GDAL/PROJ + builds
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      gdal-bin libgdal-dev proj-bin libproj-dev \
      build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

# install uv + deps (pin rasterio so it builds against libgdal-dev)
RUN pip install --no-cache-dir uv \
 && uv pip install --system --upgrade pip \
 && uv pip install --system "rasterio==1.4.3" \
 && uv pip install --system .   # installs your CLI

