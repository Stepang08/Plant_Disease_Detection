FROM python:3.11-slim

WORKDIR /app

# Install only inference dependencies (not training/dev deps).
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
    numpy

# Copy application code.
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY models/best_model.pth models/best_model.pth

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
