# NeuroXray (Chest X-ray + Brain MRI) Dual-Mode Flask App

NeuroXray is a local Flask web app that provides two AI-assisted screening modules:

- **Chest X-ray**: two lightweight PyTorch models (LiquidNN) for pneumonia / lung opacity-style classification + an occlusion-based heatmap.
- **Brain MRI**: a PyTorch **2D CNN classifier** (ResNet18) for brain tumor screening + a **Grad-CAM** localization heatmap.

This project is intended for educational / prototyping use. It is **not** a medical device.

## Features

- Clean UI for **single-scan** and **batch directory** uploads (browser `webkitdirectory`).
- **Lazy model loading** (models load on first request and are cached in memory).
- **Batch reports** exported as CSV (stored temporarily under `static/reports/`).
- Explainability:
  - Chest: occlusion sensitivity heatmap.
  - Brain: Grad-CAM heatmap (ResNet18 `layer4[-1].conv2`).
- Automatic cleanup of temporary files (reports/heatmaps) older than ~30 minutes.

## Tech Stack

- Python + Flask (server)
- PyTorch + TorchVision (models)
- PIL + OpenCV + NumPy (image handling)
- HTML/CSS templates (no React)

## Project Structure

```
.
├─ app.py
├─ models/
│  ├─ chest/
│  │  ├─ liquid_model.pth
│  │  └─ best_model.pth
│  └─ brain/
│     ├─ brain_tumor_resnet18.pth
│     ├─ best_brain_tumor_resnet18_finetuned.pth
│     └─ brain_classifier_config.json         (optional)
├─ static/
│  ├─ reports/                                (auto-created)
│  └─ heatmaps/                               (auto-created)
├─ templates/
│  ├─ main_home.html
│  ├─ error.html
│  ├─ chest/...
│  └─ brain/...
├─ check_brain_classifier.py
└─ debug_brain_classifier_images.py
```

## Requirements

This repo currently does not ship a pinned `requirements.txt`. Install the core deps:

```bash
pip install flask pillow numpy opencv-python matplotlib
pip install torch torchvision
```

Notes:
- Use a Torch build that matches your machine (CPU vs CUDA).
- Python 3.10+ is recommended.

## Run Locally

```bash
python app.py
```

Open:
- `http://127.0.0.1:5000/`

The server runs with `debug=True` by default (see `app.py` bottom). Do not use debug mode in production.

## Models (What Files Are Expected)

### Chest Models
Expected paths:
- `models/chest/liquid_model.pth`
- `models/chest/best_model.pth`

These are loaded by `load_chest_models()` and used for:
- Single: `/chest/upload_single`
- Batch: `/chest/upload_bulk`

### Brain Model (Recommended: ResNet18 Classifier)
Brain routes use `load_brain_classifier()` and expect one of:
- `models/brain/brain_tumor_resnet18.pth` (ResNet18 with `fc: Linear(512 -> 4)`)
- `models/brain/best_brain_tumor_resnet18_finetuned.pth` (ResNet18 with a bigger head `fc: 512 -> 512 -> 4`)

The default class order is:
`["glioma", "meningioma", "no tumor", "pituitary"]`

If your checkpoint uses a different class order, configure it (see below), otherwise you can get “Normal ↔ Glioma” style label swaps.

### (Legacy) Brain UNet / BraTS Note
`app.py` still contains a 2D ResUNet implementation and conversion helpers for a 3D BraTS-like checkpoint, but the **current Brain UI routes are wired to the ResNet18 classifier**.

## Brain Classifier Config (Fix Wrong Labels / Overconfidence)

Create `models/brain/brain_classifier_config.json` to override model selection and preprocessing.

Example:
```json
{
  "model_file": "best_brain_tumor_resnet18_finetuned.pth",
  "class_names": ["glioma", "meningioma", "no tumor", "pituitary"],
  "mean": [0.485, 0.456, 0.406],
  "std":  [0.229, 0.224, 0.225],
  "temperature": 2.0,
  "tumor_threshold": 0.50,
  "grayscale_3ch": false,
  "invert": false
}
```

What the knobs do:
- `model_file`: force a specific `.pth`.
- `candidate_filenames`: list of filenames to try in order (advanced).
- `class_names`: fixes class index mapping (very common source of “Normal ↔ Glioma” issues).
- `mean/std`: must match training preprocessing.
- `temperature`: softmax temperature (values > 1 reduce “100% confidence” saturation).
- `tumor_threshold`: detection threshold using `tumor_prob = 1 - P(no tumor)`.
- `grayscale_3ch`: converts image to grayscale then back to 3ch before normalization.
- `invert`: inverts intensities (sometimes needed for MRI screenshots vs training images).

## Debug / Sanity Scripts

### 1) Check That The Model Loads
```bash
python check_brain_classifier.py
```

This prints:
- which `.pth` was loaded
- the classifier head type (Linear vs Sequential)
- dummy input/output shapes

### 2) Compare Preprocessing Variants On Real Images
```bash
python debug_brain_classifier_images.py path\\to\\normal.jpg path\\to\\glioma.jpg
```

This prints top probabilities for multiple preprocessing variants (ImageNet vs medical `0.5/0.5`, invert, etc.).
Use it to diagnose:
- wrong `class_names` order
- wrong normalization
- need for `invert` / `grayscale_3ch`

## HTTP Routes (Quick Reference)

- `/` Home
- Chest:
  - `/chest`
  - `/chest/upload_single`
  - `/chest/upload_bulk`
  - `/chest/loading/<batch_id>`
  - `/chest/result_bulk/<batch_id>`
- Brain:
  - `/brain`
  - `/brain/upload_single`
  - `/brain/upload_bulk`
  - `/brain/loading/<batch_id>`
  - `/brain/result_bulk/<batch_id>`
- CSV download:
  - `/reports/download/<module>/<batch_id>`

## Batch Reports

Batch uploads generate CSV files saved under `static/reports/`:

- Chest: `chest_batch_<id>.csv`
- Brain: `brain_batch_<id>.csv`

The app deletes temp files older than `TEMP_FILE_MAX_AGE` (~30 minutes) on each request.

## Troubleshooting

### “RuntimeError: size mismatch … loading state_dict”
- Your model checkpoint does not match the model architecture created in code.
- Fix by using a **2D classifier checkpoint** compatible with ResNet18, or update the code to match your model’s architecture.

### “Normal is predicted as Glioma (and vice versa)”
Most common causes:
- `class_names` order is wrong for your checkpoint.
- preprocessing (mean/std, invert, grayscale) differs from training.

Fix:
- set correct `class_names` in `models/brain/brain_classifier_config.json`
- try `invert=true` or `grayscale_3ch=true`

### Confidence always shows 100%
Softmax can saturate (especially on out-of-distribution inputs).
Fix:
- set `temperature` > 1.0 in `brain_classifier_config.json` (e.g., `2.0`)
- ensure your input images match the dataset the model was trained on (same style, cropping, contrast)

## Production Notes

- Disable Flask debug mode.
- Use a proper WSGI server (e.g. Waitress on Windows).
- Add upload limits / authentication if running on a network.

## Disclaimer

This project provides AI-assisted outputs for experimentation only. Always consult qualified medical professionals for clinical decisions.

