FROM gemneye/seedvr-runpod:latest

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "/app/handler.py"]
