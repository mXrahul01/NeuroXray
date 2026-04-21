import os
import sys

import torch
import torch.nn as nn
from torchvision import models


CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
CANDIDATE_PATHS = [
    os.path.join("models", "brain", "brain_tumor_resnet18.pth"),
    os.path.join("models", "brain", "best_brain_tumor_resnet18_finetuned.pth"),
]


def _load_state_dict(path: str) -> dict:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            sd = obj.get(k)
            if isinstance(sd, dict):
                obj = sd
                break

    if not isinstance(obj, dict) or not all(torch.is_tensor(v) for v in obj.values()):
        raise RuntimeError("Unsupported checkpoint format; expected a PyTorch state_dict.")

    if obj and all(isinstance(k, str) and k.startswith("module.") for k in obj.keys()):
        obj = {k[len("module."):]: v for k, v in obj.items()}

    return obj


def main() -> int:
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)

    if not model_path or not os.path.exists(model_path):
        print("Missing brain classifier model file.")
        print("Expected one of:")
        for p in CANDIDATE_PATHS:
            print(" -", p)
        print("Or run: python check_brain_classifier.py <path-to-model.pth>")
        return 1

    sd = _load_state_dict(model_path)
    w = sd.get("conv1.weight")
    print("conv1.weight:", None if w is None else tuple(w.shape))
    if w is None or getattr(w, "dim", lambda: 0)() != 4:
        print("Warning: expected a 2D Conv2d weight (4D). This may be a wrong/3D checkpoint.")

    model = models.resnet18(weights=None)
    in_features = int(model.fc.in_features)
    if "fc.weight" in sd:
        out_features = int(sd["fc.weight"].shape[0])
        model.fc = nn.Linear(in_features, out_features)
        head_desc = f"fc=Linear({in_features}->{out_features})"
    elif "fc.0.weight" in sd and "fc.3.weight" in sd:
        hidden = int(sd["fc.0.weight"].shape[0])
        out_features = int(sd["fc.3.weight"].shape[0])
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_features),
        )
        head_desc = f"fc=Sequential({in_features}->{hidden}->{out_features})"
    else:
        print("Unsupported checkpoint head. Expected `fc.weight` or (`fc.0.weight` + `fc.3.weight`).")
        return 1

    print("model_path:", model_path)
    print("head:", head_desc)
    if out_features != len(CLASS_NAMES):
        print(f"Warning: checkpoint outputs {out_features} classes, but CLASS_NAMES has {len(CLASS_NAMES)}.")
    model.load_state_dict(sd, strict=True)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print("dummy input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
