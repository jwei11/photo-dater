## Photo Dater

# Can a model date a photograph?

Training a CNN to predict the decade a photo was taken — purely from visual features — then using Grad-CAM to examine whether it's learning historically meaningful signals or just artifacts.

---

## The question

Photographs carry visual signatures of their era: film grain, lighting ratios, composition conventions, clothing, and technology in the frame. This project asks whether a neural network can recover those signals well enough to assign a decade to an unseen photo — and more importantly, *what* it's actually looking at when it does.

## Pipeline overview

```
LOC API → raw images → preprocessing → EfficientNet fine-tuning → evaluation → Grad-CAM analysis → Gradio demo
```

| Stage | Details |
|---|---|
| Data | ~7,500 dated photographs from the Library of Congress (1850s–1990s) |
| Model | EfficientNet-B3, pretrained on ImageNet, fine-tuned for decade classification |
| Evaluation | Per-decade F1, confusion matrix, mean decade error |
| Interpretability | Grad-CAM heatmaps with qualitative historical analysis |
| Demo | Gradio app: upload a photo, get a predicted decade + heatmap |

## Why transfer learning (not scratch)?

Training from scratch on ~7,500 images would primarily teach a model to overfit. EfficientNet's pretrained low-level features — edges, textures, grain patterns — transfer directly to the kinds of visual signals that encode era. The interesting ML engineering is in *what* gets fine-tuned, *how* the data is balanced across decades, and *what* the model attends to — not in reinventing a backbone.

## Repository structure

```
photo-dater/
├── data/
│   ├── raw/              # Downloaded LOC images, organized by decade
│   └── processed/        # Grayscale, resized 224×224
├── src/
│   ├── scraper.py        # Library of Congress API ingestion
│   ├── preprocess.py     # Cleaning, grayscale conversion, resizing
│   ├── dataset.py        # PyTorch Dataset class
│   ├── train.py          # Fine-tuning pipeline
│   ├── evaluate.py       # Metrics, confusion matrix
│   └── gradcam.py        # Grad-CAM implementation + qualitative analysis
├── notebooks/
│   ├── 01_eda.ipynb      # Decade distribution, image quality analysis
│   └── 02_results.ipynb  # Evaluation results and Grad-CAM findings
├── app.py                # Gradio demo
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart

```bash
# Clone and set up environment
git clone https://github.com/yourusername/photo-dater.git
cd photo-dater
python -m venv photo-dating-env
source photo-dating-env/bin/activate  # Windows: photo-dating-env\Scripts\activate
pip install -r requirements.txt

# Collect data
python src/scraper.py

# Preprocess
python src/preprocess.py

# Train
python src/train.py

# Run the demo
python app.py
```

## Key design decisions

**Grayscale conversion** — all images are converted to grayscale before training. Without this, the model would learn "color photo = recent" as a trivial shortcut, bypassing the visual features we actually care about.

**Decade balancing** — pre-1900 photos are significantly underrepresented in the LOC collection. The training pipeline applies weighted sampling to prevent the model from simply predicting 1940s–1970s for everything.

**Decade classification over regression** — framing this as 15-class classification (rather than predicting a year) makes evaluation cleaner and confusion matrices more interpretable. Mean decade error is reported as a secondary metric.

## Results

*To be updated as training completes.*

| Metric | Value |
|---|---|
| Overall accuracy | — |
| Mean decade error | — |
| Hardest decade pair | — |

## Grad-CAM findings

*To be updated after interpretability analysis.*

The most interesting output of this project isn't the accuracy number — it's whether the model's attention maps correspond to historically meaningful features (clothing, technology, architectural style) or spurious correlates (scan artifacts, border styles, image degradation). That analysis lives in `notebooks/02_results.ipynb`.

## AWS infrastructure

Training runs on an EC2 `g4dn.xlarge` spot instance (~$0.16/hr). Raw images and model checkpoints are stored in S3. Experiment tracking via SageMaker.

## What's next

- Add Europeana and NYPL as additional data sources to improve pre-1900 coverage
- Experiment with regression formulation (predict year, not decade)
- Audit model errors for systematic bias toward certain geographic regions

## License

MIT — see [LICENSE](LICENSE)