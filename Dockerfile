FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    torchvision --index-url https://download.pytorch.org/whl/cpu \
    timm \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    Pillow \
    scikit-learn \
    pyyaml \
    numpy \
    huggingface-hub

COPY src/ src/
COPY api/ api/
COPY configs/ configs/
RUN mkdir -p models

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
