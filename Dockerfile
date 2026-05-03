FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python - <<'PY'
from pathlib import Path

src = Path("requirements.txt").read_text().splitlines()
filtered = [
    line
    for line in src
    if not line.startswith(
        ("torch==", "torchvision==", "torchaudio==", "streamlit", "opencv-python")
    )
]
Path("requirements-docker.txt").write_text("\n".join(filtered) + "\n")
PY

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    rm requirements-docker.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
