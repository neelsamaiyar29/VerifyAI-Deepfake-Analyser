FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 && \
    pip install --no-cache-dir \
        decord==0.6.0 \
        fastapi \
        "numpy<2" \
        pillow \
        python-multipart \
        uvicorn

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
