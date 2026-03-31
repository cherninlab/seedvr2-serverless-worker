FROM gemneye/seedvr-runpod:latest

SHELL ["/bin/bash", "-lc"]

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN set -euo pipefail; \
    export DEBIAN_FRONTEND=noninteractive; \
    PY_BIN="$(command -v python3 || command -v python || true)"; \
    if [[ -z "${PY_BIN}" ]]; then \
      apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*; \
      PY_BIN="$(command -v python3)"; \
    fi; \
    if ! "${PY_BIN}" -m pip --version >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends git ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*; \
    "${PY_BIN}" -m pip install --break-system-packages --no-cache-dir setuptools wheel; \
    "${PY_BIN}" -m pip install --break-system-packages --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision; \
    git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler /opt/seedvr2_videoupscaler; \
    "${PY_BIN}" -c "from pathlib import Path; req=Path('/opt/seedvr2_videoupscaler/requirements.txt').read_text(encoding='utf-8').splitlines(); filtered=[line for line in req if line.strip() and not line.startswith(('torch','torchvision'))]; Path('/tmp/seedvr2_requirements_no_torch.txt').write_text('\\n'.join(filtered)+'\\n', encoding='utf-8')"
RUN set -euo pipefail; \
    PY_BIN="$(command -v python3 || command -v python)"; \
    "${PY_BIN}" -m pip install --break-system-packages --no-cache-dir -r /tmp/seedvr2_requirements_no_torch.txt; \
    "${PY_BIN}" -m pip install --break-system-packages --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY sitecustomize.py /app/sitecustomize.py

ENV PYTHONUNBUFFERED=1
ENV SEEDVR2_APP_ROOT=/opt/seedvr2_videoupscaler

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["PY_BIN=\"$(command -v python3 || command -v python)\"; exec \"${PY_BIN}\" -u /app/handler.py"]
