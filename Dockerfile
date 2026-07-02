FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev curl \
    libgl1 libglib2.0-0 ffmpeg \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    python3.11 -m pip install --no-cache-dir -q -r requirements.txt

COPY src/ ./src/
COPY main.py .
COPY cloud_run.py .

ENTRYPOINT ["python", "cloud_run.py"]
