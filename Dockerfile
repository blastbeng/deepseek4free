FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    DOCKERMODE=true

# Chromium + Xvfb (Cloudflare bypass via DrissionPage needs a display even headless)
# Debian's chromium is used because Google's chrome repo does not resolve on
# Debian trixie (time64 shared-lib variants are incompatible with the deb).
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium chromium-driver xvfb xauth dbus ca-certificates curl \
    fonts-liberation fonts-noto-color-emoji \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dsk/ ./dsk/

# Persistent data: cookies.json is written here (mount a volume)
ENV COOKIES_DIR=/data
RUN mkdir -p /data

EXPOSE 8000

# Run the OpenAI-compatible server (serves /v1/chat/completions, /v1/models)
CMD ["python", "-m", "dsk.openai_server"]