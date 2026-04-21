import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


CLASS_NAMES_DEFAULT = ["glioma", "meningioma", "no tumor", "pituitary"]
MODEL_FILES = [
    os.path.join("models", "brain", "brain_tumor_resnet18.pth"),
    os.path.join("models", "brain", "best_brain_tumor_resnet18_finetuned.pth"),
]


def load_state_dict(path: str) -> dict:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            sd = obj.get(k)
            if isinstance(sd, dict) and sd and all(torch.is_tensor(v) for v in sd.values()):
                obj = sd
                break

    if not isinstance(obj, dict) or not obj or not all(torch.is_tensor(v) for v in obj.values()):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    if all(isinstance(k, str) and k.startswith("module.") for k in obj.keys()):
        obj = {k[len("module."):]: v for k, v in obj.items()}

    return obj


def build_model_from_state_dict(sd: dict) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = int(model.fc.in_features)
    if "fc.weight" in sd:
        out_features = int(sd["fc.weight"].shape[0])
        model.fc = nn.Linear(in_features, out_features)
    elif "fc.0.weight" in sd and "fc.3.weight" in sd:
        hidden = int(sd["fc.0.weight"].shape[0])
        out_features = int(sd["fc.3.weight"].shape[0])
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_features),
        )
    else:
        raise RuntimeError("Unsupported head. Expected `fc.weight` or (`fc.0.weight` + `fc.3.weight`).")

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def preprocess(pil_img: Image.Image, mean: List[float], std: List[float], invert: bool) -> torch.Tensor:
    img = pil_img.convert("RGB")
    if invert:
        img = ImageOps.invert(img)
    t = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return t(img).unsqueeze(0)


def topk(probs: torch.Tensor, k: int = 4) -> List[Tuple[int, float]]:
    v, i = torch.topk(probs, k=min(k, probs.numel()))
    return [(int(ii.item()), float(vv.item())) for vv, ii in zip(v, i)]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python debug_brain_classifier_images.py <image1> [image2 ...]")
        return 1

    images = sys.argv[1:]
    for path in images:
        if not os.path.exists(path):
            print("Missing image:", path)
            return 1

    model_paths = [p for p in MODEL_FILES if os.path.exists(p)]
    if not model_paths:
        print("No model files found. Expected one of:")
        for p in MODEL_FILES:
            print(" -", p)
        return 1

    variants = [
        ("imagenet", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False),
        ("medical_0.5", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False),
        ("imagenet_invert", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
    ]

    for model_path in model_paths:
        sd = load_state_dict(model_path)
        model = build_model_from_state_dict(sd)
        print("\n=== MODEL:", os.path.basename(model_path), "===\n")

        for img_path in images:
            pil = Image.open(img_path)
            pil = ImageOps.exif_transpose(pil)
            print("image:", img_path)
            for name, mean, std, invert in variants:
                x = preprocess(pil, mean, std, invert)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                tk = topk(probs, 4)
                line = []
                for idx, p in tk:
                    label = CLASS_NAMES_DEFAULT[idx] if idx < len(CLASS_NAMES_DEFAULT) else str(idx)
                    line.append(f"{label}={p*100:.2f}%")
                print(f"  {name:14s} " + " | ".join(line))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

