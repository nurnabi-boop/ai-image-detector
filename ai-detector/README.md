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

## Step 2 — train

Two-phase schedule: the backbone is frozen for `--epochs-frozen` epochs while
the head trains, then the whole network fine-tunes for `--epochs-unfrozen`
epochs (lower backbone LR, cosine decay, early stopping on val ROC-AUC). The
best checkpoint goes to `models/best.pt`, and test-set metrics and plots go to
`reports/`.

```bash
# quick run on a subset
python -m src.train --max-samples 5000 --epochs-frozen 1 --epochs-unfrozen 4

# full run, ConvNeXt-Tiny, logged to Weights & Biases
python -m src.train --backbone convnext_tiny --batch-size 64 --wandb
```

You'll want a GPU (Colab / Kaggle are fine). AMP is on automatically on CUDA.
To check that everything runs without downloading anything:

```bash
python -m src.train --synthetic 128 --backbone efficientnet_b0 --image-size 96 \
    --no-pretrained --epochs-frozen 1 --epochs-unfrozen 1 --num-workers 0
```

Any timm CNN works as `--backbone`. ViT-style backbones won't work with
Grad-CAM, because it needs a spatial feature map.

## Step 3 — evaluate

Rebuilds the same test split (same dataset, `--max-samples` and seed as in
training) and writes `metrics.json`, `confusion_matrix.png` and `roc_curve.png`.
The AI-generated class is treated as positive.

```bash
python -m src.evaluate --checkpoint models/best.pt --out-dir reports
```

## Step 4 — Grad-CAM

```bash
python -m src.gradcam --checkpoint models/best.pt --image some_image.jpg
# -> some_image_gradcam.png
```

## Step 5 — Gradio demo

```bash
python app.py --checkpoint models/best.pt          # http://localhost:7860
python app.py --checkpoint models/best.pt --share  # public link
```

## Step 6 — ONNX export

Exports with a dynamic batch dimension, stores class names, input size and
normalization in the ONNX metadata, and checks outputs against PyTorch.

```bash
python -m src.export_onnx --checkpoint models/best.pt --out models/detector.onnx
```

## Tests

CPU-only, no downloads (tiny untrained backbone + synthetic images):

```bash
pip install pytest
pytest -q
```

## Project layout

```
ai-detector/
├── src/
│   ├── dataset.py      # HF loader, augmentations, splits, synthetic data
│   ├── model.py        # timm backbone + head, freeze/unfreeze, checkpoints
│   ├── train.py        # frozen→unfrozen schedule, W&B, early stop
│   ├── evaluate.py     # metrics, confusion matrix, ROC
│   ├── gradcam.py      # heatmap overlay
│   └── export_onnx.py  # ONNX export + parity check
├── tests/
├── models/             # checkpoints (git-ignored)
├── reports/            # metrics + plots (git-ignored)
├── app.py              # Gradio demo
├── requirements.txt
└── README.md
```
