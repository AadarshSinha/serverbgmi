# syntax=docker/dockerfile:1
#
# Two stages: build the virtualenv with a compiler available, then copy just
# the finished venv into a clean runtime image. The result is reproducible —
# none of the manual `pip uninstall torch` / `apt install libgl1` steps from
# the old runbook are needed, because they are baked in below.

# ---------------------------------------------------------------- builder ---
FROM python:3.10-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Every pinned dependency ships a manylinux wheel today, so nothing actually
# compiles. build-essential is insurance: a future version bump that needs a
# compiler should fail loudly here rather than at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# torch goes in FIRST, from PyTorch's CPU-only index. The default PyPI wheel
# drags in ~2GB of CUDA libraries that are dead weight on a CPU droplet — this
# is the containerised version of the uninstall-and-reinstall dance in the old
# runbook, except the CUDA build is never downloaded in the first place.
RUN pip install torch==2.2.2 torchvision==0.17.2 \
      --index-url https://download.pytorch.org/whl/cpu

# requirements.txt pins those same two versions, so pip sees them satisfied and
# leaves the CPU builds alone while installing everything else from PyPI.
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.10-slim-bookworm

# cv2 dlopens libGL and glib at import time — this is `apt install -y libgl1
# libglib2.0-0` from the old runbook, plus two libs opencv reaches for on some
# code paths and libgomp for scikit-learn.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# YOLO_CONFIG_DIR / MPLCONFIGDIR: ultralytics and matplotlib want a writable
# home directory, and the app does not run as root.
# LOCAL_UPLOAD_DIR: uploads (and, in development, the SQLite file) live on the
# mounted volume, never inside an image layer.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=4000 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    LOCAL_UPLOAD_DIR=/data/uploads

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app /data/uploads \
 && chown -R appuser:appuser /app /data

WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 4000

# Importing TensorFlow and loading 14 Keras models plus the YOLO detector takes
# the better part of a minute, so the start period is deliberately long: a
# container that is still booting must not be reported unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:$PORT/health',timeout=4).status==200 else 1)"

# `app:create_app()` is an application factory — `app:app` does not exist and
# will not start. exec keeps gunicorn as PID 1 so it receives SIGTERM directly.
#
# ${GUNICORN_RELOAD:+--reload} adds the flag only when the variable is set, so
# one image serves both environments: docker-compose.dev.yml sets it, production
# leaves it unset and the word disappears entirely.
CMD ["sh", "-c", "exec gunicorn 'app:create_app()' --bind 0.0.0.0:${PORT:-4000} --workers ${GUNICORN_WORKERS:-1} --timeout ${GUNICORN_TIMEOUT:-120} --graceful-timeout 30 ${GUNICORN_RELOAD:+--reload} --access-logfile - --error-logfile -"]
