FROM gemneye/seedvr-runpod:latest

SHELL ["/bin/bash", "-lc"]

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN set -euo pipefail; \
    PY_BIN="$(command -v python3 || command -v python || true)"; \
    if [[ -z "${PY_BIN}" ]]; then \
      apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*; \
      PY_BIN="$(command -v python3)"; \
    fi; \
    "${PY_BIN}" -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash", "-lc", "PY_BIN=\"$(command -v python3 || command -v python)\"; exec \"${PY_BIN}\" -u /app/handler.py"]
