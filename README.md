# Plant Disease Detection

Classifies plant diseases from leaf images. Outputs **disease-only labels** (e.g., "late blight") regardless of host plant species.

## Quick Start

```bash
# Clone
git clone https://github.com/Stepang08/Plant_Disease_Detection.git
cd Plant_Disease_Detection

# Install
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload
# Open http://localhost:8000/docs for Swagger UI
# Model weights are downloaded automatically from HuggingFace Hub on first run.
```

## Project Structure

```
├── src/
│   ├── dataset.py       # Label mapping (82 folders → 39 diseases), stratified split
│   ├── transforms.py    # Train/val/TTA augmentation pipelines
│   ├── model.py         # Model factory (EfficientNet via timm, DINOv2 via torch.hub)
│   ├── train.py         # Training loop with transfer learning and W&B logging
│   ├── evaluate.py      # mAP evaluation, per-class AP, confusion matrix
│   ├── predict.py       # Single-image inference with optional TTA
│   └── utils.py         # Seeding, config loading, metrics
├── api/
│   ├── main.py          # FastAPI: POST /predict, GET /health, GET /classes
│   └── schemas.py       # Pydantic request/response models
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis (rendered with plots)
│   ├── 02_training.ipynb         # EfficientNet training (Colab-ready)
│   ├── 03_dinov2_training.ipynb  # DINOv2 ViT-B/14 training
│   └── 04_dinov2_vitl14_training.ipynb  # DINOv2 ViT-L/14 training (final model)
├── configs/
│   ├── default.yaml     # Training hyperparameters
│   └── label_mapping.json # 39 disease class definitions
├── models/              # Saved model weights (downloaded from HF Hub)
├── reports/             # Technical report (PDF)
├── tests/               # API tests
├── Dockerfile           # For HuggingFace Spaces deployment
└── requirements.txt
```

## Dataset

~8,000 images across 82 plant-disease folder classes, merged into **39 disease-only classes** by stripping plant species prefixes. Key challenge: the model must learn plant-agnostic disease features (e.g., "rust" looks similar on wheat, corn, and soybean).

**Label mapping:** `src/dataset.py` → `configs/label_mapping.json`

## Approach

- **Final model:** DINOv2 (ViT-L/14) frozen backbone + linear head (40K trainable params, 0.897 mAP)
- **Baseline:** EfficientNet-B0 fine-tuned (4.06M trainable params, 0.66 mAP)
- **Key insight:** Foundation model features with minimal training outperform full fine-tuning by +24pp mAP while using 100× fewer trainable parameters
- **Class imbalance:** Inverse-frequency weighted sampler (fine-tuning only)
- **Metric:** mAP (mean Average Precision) with per-class AP tracking

## Experiments

All experiments tracked in Weights & Biases:

**[W&B Project Dashboard](https://wandb.ai/stepan-goyunyan-physmath-school-after-a-shahinyan-/plant-disease-detection)**

| Run | Backbone | Trainable Params | Val mAP | Notes |
|-----|----------|--------|---------|-------|
| Baseline | EfficientNet-B0 | 4.06M | 0.651 | Default hyperparameters |
| Tuned B0 | EfficientNet-B0 | 4.06M | 0.656 | Higher backbone LR (1e-4) |
| B3 | EfficientNet-B3 | 10.76M | 0.622 | Overfits — val loss unstable |
| DINOv2 B/14 | ViT-B/14 + linear | 30K | 0.863 | Frozen backbone, linear head |
| **DINOv2 L/14** | **ViT-L/14 + linear** | **40K** | **0.897** | **Final model** |

## API

FastAPI endpoint with Swagger documentation.

**Live API:** https://stepang08-plant-disease-detection.hf.space/docs

**Model weights:** https://huggingface.co/Stepang08/plant-disease-model

```bash
# Local
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Docker
docker build -t plant-disease .
docker run -p 7860:7860 plant-disease
```

**Endpoints:**
- `POST /predict` — Upload an image, get disease prediction + confidence
- `GET /health` — Health check
- `GET /classes` — List all 39 disease classes

## Training (Colab)

Training notebooks are designed for Google Colab with a T4 GPU runtime:

- `notebooks/02_training.ipynb` — EfficientNet fine-tuning
- `notebooks/03_dinov2_training.ipynb` — DINOv2 ViT-B/14 (0.86 mAP)
- `notebooks/04_dinov2_vitl14_training.ipynb` — DINOv2 ViT-L/14 (0.90 mAP, final model)

## Report

See `reports/report.pdf` for the full technical report covering methodology, ablation studies, and results.
