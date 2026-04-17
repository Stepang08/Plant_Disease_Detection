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
```

## Project Structure

```
├── src/
│   ├── dataset.py       # Label mapping (82 folders → 39 diseases), stratified split
│   ├── transforms.py    # Train/val/TTA augmentation pipelines
│   ├── model.py         # Model factory (timm-based EfficientNet, ConvNeXt, ResNet)
│   ├── train.py         # Training loop with transfer learning and W&B logging
│   ├── evaluate.py      # mAP evaluation, per-class AP, confusion matrix
│   ├── predict.py       # Single-image inference with optional TTA
│   └── utils.py         # Seeding, config loading, metrics
├── api/
│   ├── main.py          # FastAPI: POST /predict, GET /health, GET /classes
│   └── schemas.py       # Pydantic request/response models
├── notebooks/
│   ├── 01_eda.ipynb     # Exploratory data analysis (rendered with plots)
│   ├── 02_training.ipynb # Colab-ready training notebook
│   └── 03_evaluation.ipynb # Model evaluation and error analysis
├── configs/
│   ├── default.yaml     # Training hyperparameters
│   └── label_mapping.json # 39 disease class definitions
├── models/              # Saved model weights
├── reports/             # Technical report (PDF)
├── tests/               # API tests
├── Dockerfile           # For HuggingFace Spaces deployment
└── requirements.txt
```

## Dataset

~8,000 images across 82 plant-disease folder classes, merged into **39 disease-only classes** by stripping plant species prefixes. Key challenge: the model must learn plant-agnostic disease features (e.g., "rust" looks similar on wheat, corn, and soybean).

**Label mapping:** `src/dataset.py` → `configs/label_mapping.json`

## Approach

- **Backbone:** EfficientNet-B0 (5.3M params) via `timm`
- **Transfer learning:** Freeze backbone for 3 epochs, then fine-tune with discriminative learning rates
- **Augmentation:** RandomResizedCrop, flips, rotation, color jitter (small hue — disease color matters)
- **Class imbalance:** Inverse-frequency weighted sampler
- **Metric:** mAP (mean Average Precision) with per-class AP tracking

## Experiments

All experiments tracked in Weights & Biases:

**[W&B Project Dashboard](https://wandb.ai/stepan-goyunyan-physmath-school-after-a-shahinyan-/plant-disease-detection)**

| Run | Backbone | Params | Val mAP | Notes |
|-----|----------|--------|---------|-------|
| Baseline | EfficientNet-B0 | 5.3M (4.06M trainable) | 0.65 | Default hyperparameters |
| Tuned B0 | EfficientNet-B0 | 5.3M (4.06M trainable) | 0.66 | Higher backbone LR (1e-4) |
| B3 | EfficientNet-B3 | 10.8M trainable | 0.62 | Overfits — val loss unstable |
| **DINOv2** | **ViT-B/14 + linear** | **30K trainable** | **0.86** | **Frozen backbone, linear head** |

## API

FastAPI endpoint with Swagger documentation.

**Live API:** https://stepang08-plant-disease-detection.hf.space/docs

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

Open `notebooks/02_training.ipynb` in Google Colab with a T4 GPU runtime. The notebook handles:
- Cloning the repo
- Mounting Google Drive for dataset access
- Installing dependencies
- Running training with W&B logging
- Saving checkpoints to Drive

## Report

See `reports/report.pdf` for the full technical report covering methodology, ablation studies, and results.
