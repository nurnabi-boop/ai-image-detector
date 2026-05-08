# AI Image Detector

Detects whether an image is real or AI-generated. PyTorch + timm backbone
(EfficientNet-B3 / ConvNeXt-Tiny), trained on a HF-hosted real-vs-AI dataset
with albumentations augmentation, W&B tracking, and a Gradio demo with
Grad-CAM heatmaps.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Step 1 — verify the data pipeline

Loads a small subset, builds train/val/test DataLoaders, prints sizes, and
saves a 16-image grid (`batch_preview.png`) of augmented training samples
with their labels.

```bash
python -m src.dataset --max-samples 200 --batch-size 16 --out batch_preview.png
```

Default dataset: `Hemg/AI-Generated-vs-Real-Images-Datasets` (HF mirror,
~150K images, smaller and easier to pull than full GenImage). To use a
different mirror:

```bash
python -m src.dataset --dataset <hf-dataset-name>
```

Label convention used downstream: **0 = Real, 1 = AI-Generated**.

## Project layout

```
ai-detector/
├── data/
├── src/
│   ├── dataset.py     # ← built; HF loader, augmentations, splits
│   ├── model.py       # next: timm backbone + classification head
│   ├── train.py       # next: frozen→unfrozen schedule, W&B, early stop
│   ├── evaluate.py    # next: metrics, confusion matrix, ROC
│   └── gradcam.py     # next: heatmap overlay
├── models/
├── notebooks/
│   └── eda.ipynb
├── app.py             # next: Gradio demo
├── requirements.txt
└── README.md
```
