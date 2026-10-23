"""Gradio demo: upload an image, get Real vs AI-Generated probabilities and a
Grad-CAM heatmap of the regions that drove the prediction.

  python app.py                              # uses models/best.pt
  python app.py --checkpoint path/to/best.pt --share
  MODEL_PATH=path/to/best.pt python app.py   # e.g. on Hugging Face Spaces
"""
from __future__ import annotations

import argparse
import os

import gradio as gr
import torch
from PIL import Image

from src.gradcam import explain
from src.model import load_checkpoint


def build_demo(checkpoint: str) -> gr.Blocks:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(checkpoint, device)

    def predict(image: Image.Image | None):
        if image is None:
            return None, None
        result = explain(model, image, device=device)
        return result.probs, result.overlay

    with gr.Blocks(title="AI Image Detector") as demo:
        gr.Markdown(
            "# AI Image Detector\n"
            "Upload an image to estimate whether it is a real photo or AI-generated. "
            "The heatmap highlights regions that drove the prediction (Grad-CAM). "
            "Predictions are probabilistic. Don't treat them as proof."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Image")
            with gr.Column():
                label = gr.Label(num_top_classes=2, label="Prediction")
                heatmap = gr.Image(type="pil", label="Grad-CAM")
        gr.Button("Analyze", variant="primary").click(predict, inp, [label, heatmap])
        inp.upload(predict, inp, [label, heatmap])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio demo.")
    parser.add_argument("--checkpoint", default=os.environ.get("MODEL_PATH", "models/best.pt"))
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_demo(args.checkpoint).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
