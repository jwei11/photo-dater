"""
Gradio demo: upload a photo, get predicted decade + Grad-CAM heatmap.

Usage:
    python app.py
    python app.py --share
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import DECADES, IDX_TO_DECADE, EVAL_TRANSFORMS
from gradcam import GradCAM, overlay_heatmap
from train import load_checkpoint

CHECKPOINT_PATH = Path("checkpoints/best.pt")


def load_demo_model(device: torch.device):
    model = load_checkpoint(CHECKPOINT_PATH, device)
    gradcam = GradCAM(model, model.blocks[-1])
    return model, gradcam


def predict(image: np.ndarray | None, model, gradcam, device):
    if image is None:
        return "Upload an image to get a decade prediction.", None

    pil_img = Image.fromarray(image).convert("L").resize((300, 300))
    tensor = EVAL_TRANSFORMS(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = logits.softmax(1)[0].cpu().numpy()

    cam, pred_idx, _confidence = gradcam(tensor)
    pred_decade = IDX_TO_DECADE[pred_idx]
    overlay = overlay_heatmap(pil_img, cam)

    top3 = probs.argsort()[-3:][::-1]
    lines = [f"**Predicted decade: {pred_decade}s** ({probs[pred_idx]:.1%} confidence)\n"]
    lines.append("Top predictions:")
    for idx in top3:
        lines.append(f"- {DECADES[idx]}s: {probs[idx]:.1%}")

    return "\n".join(lines), overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model, gradcam = load_demo_model(device)

    def _predict(image):
        return predict(image, model, gradcam, device)

    with gr.Blocks(title="Photo Dater") as demo:
        gr.Markdown(
            "# Photo Dater\n"
            "Upload a photograph to predict which decade it was taken in, "
            "with a Grad-CAM heatmap showing where the model is looking."
        )
        with gr.Row():
            inp = gr.Image(type="numpy", label="Upload photo")
            out_img = gr.Image(type="numpy", label="Grad-CAM overlay")
        out_text = gr.Markdown(label="Prediction")
        btn = gr.Button("Predict")
        btn.click(_predict, inp, [out_text, out_img])
        inp.change(_predict, inp, [out_text, out_img])

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
