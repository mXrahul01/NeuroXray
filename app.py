"""
NeuroXray - Dual-Mode AI Medical Diagnostic Platform
====================================================
Modules: Chest X-ray (LNN) & Brain MRI (CNN Classification + Grad-CAM)
Architecture: Pure Flask & PyTorch (Zero TensorFlow)
Standards: Zero Permanent Storage, Lazy Model Loading, Explainable AI Heatmaps.
"""

import os
import json
import io
import csv
import uuid
import time
import base64
import logging
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# Pure PyTorch Stack
import torch
import torch.nn as nn
from torchvision import transforms, models

from flask import (
    Flask, request, redirect, url_for, render_template,
    jsonify, abort, send_file
)
from werkzeug.utils import secure_filename

# ==============================================================================
# CONFIGURATION & LIFECYCLE MANAGEMENT
# ==============================================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB max for bulk uploads
app.config["JSON_SORT_KEYS"] = False

REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
HEATMAP_DIR = os.path.join(BASE_DIR, "static", "heatmaps")
MODEL_CHEST_DIR = os.path.join(BASE_DIR, "models", "chest")
MODEL_BRAIN_DIR = os.path.join(BASE_DIR, "models", "brain")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
TEMP_FILE_MAX_AGE = 1800  # 30 mins limit

for d in [REPORT_DIR, HEATMAP_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BrainModelLoadError(RuntimeError):
    pass

def cleanup_old_temp_files():
    current_time = time.time()
    for directory in [REPORT_DIR, HEATMAP_DIR]:
        if not os.path.exists(directory): continue
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if current_time - os.path.getmtime(filepath) > TEMP_FILE_MAX_AGE:
                    try: os.remove(filepath)
                    except Exception: pass

@app.before_request
def before_request_hook():
    cleanup_old_temp_files()

# ==============================================================================
# MODEL ARCHITECTURES: CHEST LNN
# ==============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return (1 - self.tau) * h_prev + self.tau * torch.tanh(self.W_in(x) + self.W_rec(h_prev))

class LiquidNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.liquid_neuron = LiquidNeuron(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = torch.zeros(b, self.liquid_neuron.hidden_size, device=x.device)
        return self.fc(self.liquid_neuron(x.view(b, -1), h))

# ==============================================================================
# MODEL ARCHITECTURES: BRAIN ResUNet (2D)
# ==============================================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + res)

class BrainResUNet(nn.Module):
    """
    Lightweight 2D ResUNet used for brain MRI slice inference.

    Note: The shipped checkpoint `models/brain/best_brats_model_dice.pth` is a 3D BraTS-style
    model (Conv3d, 4 input channels, 4 output channels). We convert those weights to 2D at load
    time by collapsing the depth kernel dimension (mean over D). This prevents runtime crashes
    while keeping the app compatible with 2D image uploads (PNG/JPG).
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super().__init__()
        # Encoder
        self.enc1 = ResBlock(in_channels, base_filters)
        self.enc2 = ResBlock(base_filters, base_filters*2)

        # Bottleneck
        self.bottleneck = ResBlock(base_filters*2, base_filters*4)

        # Decoder 2
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base_filters*4, base_filters*2) # Due to concat

        # Decoder 1
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base_filters*2, base_filters) # Due to concat

        # Output
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)

# ==============================================================================
# GLOBAL CACHES & LAZY LOADING
# ==============================================================================
chest_cache = {"m1": None, "m2": None}
brain_cache = {"model": None, "spec": None}
brain_classifier_cache = {"model": None, "model_file": None}

BRAIN_CLASSIFIER_CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
BRAIN_CLASSIFIER_CONFIG_PATH = os.path.join(MODEL_BRAIN_DIR, "brain_classifier_config.json")

def load_brain_classifier_config() -> Dict[str, Any]:
    if not os.path.exists(BRAIN_CLASSIFIER_CONFIG_PATH):
        return {}
    try:
        with open(BRAIN_CLASSIFIER_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            logging.warning("Brain classifier config is not a JSON object: %s", BRAIN_CLASSIFIER_CONFIG_PATH)
            return {}
        return cfg
    except Exception as e:
        logging.warning("Failed to read brain classifier config (%s): %s", BRAIN_CLASSIFIER_CONFIG_PATH, e)
        return {}

def _torch_load_weights(path: str) -> Dict[str, torch.Tensor]:
    # Safe-load tensors only (PyTorch 2.0+).
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            sd = obj.get(k)
            if isinstance(sd, dict):
                return sd
        # Plain state_dict case (most common): all values are tensors.
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
        raise BrainModelLoadError("Unsupported brain checkpoint format (expected a state_dict).")

    raise BrainModelLoadError("Unsupported brain checkpoint format (expected a state_dict).")

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def _infer_brain_checkpoint_spec(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    w_in = state_dict.get("enc1.conv1.weight")
    w_out = state_dict.get("out_conv.weight")
    if w_in is None or w_out is None:
        raise BrainModelLoadError("Brain checkpoint missing expected keys (enc1.conv1.weight/out_conv.weight).")

    if w_in.dim() == 4:
        conv_dim = "2d"
        in_channels = int(w_in.shape[1])
        base_filters = int(w_in.shape[0])
    elif w_in.dim() == 5:
        conv_dim = "3d"
        in_channels = int(w_in.shape[1])
        base_filters = int(w_in.shape[0])
    else:
        raise BrainModelLoadError(f"Unsupported brain checkpoint conv weight dims: {w_in.dim()}.")

    out_channels = int(w_out.shape[0])
    return {"conv_dim": conv_dim, "in_channels": in_channels, "out_channels": out_channels, "base_filters": base_filters}

def _convert_3d_state_dict_to_2d(state_dict_3d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}
    for k, v in state_dict_3d.items():
        if not torch.is_tensor(v):
            continue
        if v.dim() == 5:
            # Conv3d/ConvTranspose3d weights: [O,I,D,H,W] or [I,O,D,H,W].
            converted[k] = v.mean(dim=2)
        else:
            converted[k] = v
    return converted

def load_chest_models():
    if chest_cache["m1"] is None or chest_cache["m2"] is None:
        logging.info("Initializing Chest LNN Models...")
        m1, m2 = LiquidNN(224*224, 128, 2).to(DEVICE), LiquidNN(224*224, 128, 2).to(DEVICE)
        m1.load_state_dict(torch.load(os.path.join(MODEL_CHEST_DIR, "liquid_model.pth"), map_location=DEVICE))
        m2.load_state_dict(torch.load(os.path.join(MODEL_CHEST_DIR, "best_model.pth"), map_location=DEVICE))
        m1.eval(); m2.eval()
        chest_cache["m1"], chest_cache["m2"] = m1, m2
    return chest_cache["m1"], chest_cache["m2"]

def load_brain_model():
    if brain_cache["model"] is None:
        logging.info("Initializing Brain ResUNet Model...")
        model_path = os.path.join(MODEL_BRAIN_DIR, "best_brats_model_dice.pth")

        if not os.path.exists(model_path):
            raise BrainModelLoadError(f"Brain model file not found: {model_path}")

        state_dict = _torch_load_weights(model_path)
        spec = _infer_brain_checkpoint_spec(state_dict)

        if spec["conv_dim"] == "3d":
            logging.warning(
                "Brain checkpoint is Conv3d (%s in-ch, %s out-ch, base=%s). Converting to 2D for PNG/JPG inference.",
                spec["in_channels"], spec["out_channels"], spec["base_filters"],
            )
            state_dict = _convert_3d_state_dict_to_2d(state_dict)
            spec = {**spec, "conv_dim": "2d"}

        model = BrainResUNet(
            in_channels=spec["in_channels"],
            out_channels=spec["out_channels"],
            base_filters=spec["base_filters"],
        )

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise BrainModelLoadError(f"Brain checkpoint is not compatible with BrainResUNet2D: {e}") from e

        model.eval()
        brain_cache["model"] = model.to(DEVICE)
        brain_cache["spec"] = spec
        logging.info("Brain ResUNet loaded successfully! (conv=%s, in=%s, out=%s, base=%s)",
                     spec["conv_dim"], spec["in_channels"], spec["out_channels"], spec["base_filters"])

    return brain_cache["model"], brain_cache["spec"]

def load_brain_classifier():
    if brain_classifier_cache["model"] is None:
        logging.info("Initializing Brain Tumor Classifier (ResNet18)...")

        cfg = load_brain_classifier_config()

        candidate_filenames = [
            # Prefer the original, known-good ResNet18 checkpoint unless config overrides it.
            "brain_tumor_resnet18.pth",
            "best_brain_tumor_resnet18_finetuned.pth",
        ]
        if isinstance(cfg.get("candidate_filenames"), list) and all(isinstance(x, str) for x in cfg["candidate_filenames"]):
            candidate_filenames = cfg["candidate_filenames"]

        # Explicit model selection wins.
        if isinstance(cfg.get("model_file"), str) and cfg["model_file"].strip():
            candidate_filenames = [cfg["model_file"].strip()]
        model_path = None
        for filename in candidate_filenames:
            p = os.path.join(MODEL_BRAIN_DIR, filename)
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            expected = ", ".join(candidate_filenames)
            raise BrainModelLoadError(f"Brain classifier model not found in {MODEL_BRAIN_DIR}. Expected one of: {expected}")

        state_dict = _strip_module_prefix(_torch_load_weights(model_path))

        model = models.resnet18(weights=None)
        in_features = int(model.fc.in_features)

        if "fc.weight" in state_dict:
            out_features = int(state_dict["fc.weight"].shape[0])
            model.fc = nn.Linear(in_features, out_features)
            head_desc = f"fc=Linear({in_features}->{out_features})"
        elif "fc.0.weight" in state_dict and "fc.3.weight" in state_dict:
            fc0_w = state_dict["fc.0.weight"]
            fc3_w = state_dict["fc.3.weight"]
            hidden = int(fc0_w.shape[0])
            out_features = int(fc3_w.shape[0])
            model.fc = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden, out_features),
            )
            head_desc = f"fc=Sequential({in_features}->{hidden}->{out_features})"
        else:
            raise BrainModelLoadError(
                "Unsupported ResNet18 classifier head in checkpoint. Expected `fc.weight` or (`fc.0.weight` + `fc.3.weight`)."
            )

        class_names = cfg.get("class_names") if isinstance(cfg.get("class_names"), list) else BRAIN_CLASSIFIER_CLASS_NAMES
        if not class_names or not all(isinstance(x, str) and x.strip() for x in class_names):
            raise BrainModelLoadError("Brain classifier class_names must be a non-empty list of strings.")

        if out_features != len(class_names):
            raise BrainModelLoadError(
                f"Brain classifier outputs {out_features} classes but class_names has {len(class_names)}."
            )

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise BrainModelLoadError(f"Brain classifier checkpoint is not compatible with ResNet18: {e}") from e

        model.eval()
        brain_classifier_cache["model"] = model.to(DEVICE)
        brain_classifier_cache["model_file"] = os.path.basename(model_path)
        logging.info("Brain classifier loaded successfully! file=%s %s", os.path.basename(model_path), head_desc)

    cfg = load_brain_classifier_config()
    class_names = cfg.get("class_names") if isinstance(cfg.get("class_names"), list) else BRAIN_CLASSIFIER_CLASS_NAMES

    mean = cfg.get("mean") if isinstance(cfg.get("mean"), list) else [0.485, 0.456, 0.406]
    std = cfg.get("std") if isinstance(cfg.get("std"), list) else [0.229, 0.224, 0.225]
    temperature = cfg.get("temperature", 1.0)
    tumor_threshold = cfg.get("tumor_threshold", 0.50)
    grayscale_3ch = bool(cfg.get("grayscale_3ch", False))
    invert = bool(cfg.get("invert", False))

    meta = {
        "arch": "resnet18",
        "class_names": class_names,
        "img_size": 224,
        "in_channels": 3,
        "mean": mean,
        "std": std,
        "temperature": temperature,
        "tumor_threshold": tumor_threshold,
        "grayscale_3ch": grayscale_3ch,
        "invert": invert,
        "model_file": brain_classifier_cache.get("model_file"),
    }
    return brain_classifier_cache["model"], meta

# ==============================================================================
# PREPROCESSING & IMAGE UTILS
# ==============================================================================
chest_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_brain_classifier_image(pil_img: Image.Image, meta: Dict[str, Any]) -> torch.Tensor:
    img_size = int(meta.get("img_size", 224))
    mean = meta.get("mean", [0.485, 0.456, 0.406])
    std = meta.get("std", [0.229, 0.224, 0.225])

    img = pil_img
    if bool(meta.get("grayscale_3ch", False)):
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")

    if bool(meta.get("invert", False)):
        img = ImageOps.invert(img)

    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return t(img)

def _normalize_brain_class_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")

def is_brain_tumor_class(name: str) -> bool:
    n = _normalize_brain_class_name(name)
    return n not in {"no tumor", "no tumour", "normal", "no_tumor", "notumor", "no tumour "}

def is_no_tumor_label(name: str) -> bool:
    n = _normalize_brain_class_name(name)
    return n in {"no tumor", "no tumour", "normal", "no_tumor", "notumor"}

def find_no_tumor_index(class_names: List[str]) -> int | None:
    for i, name in enumerate(class_names):
        if is_no_tumor_label(name):
            return i
    return None

def format_brain_class_label(name: str) -> str:
    n = _normalize_brain_class_name(name)
    if n in {"no tumor", "no tumour", "normal", "no_tumor", "notumor"}:
        return "Normal"
    # Keep type, but make it look like a diagnosis label.
    return n.title()

def preprocess_brain_image(pil_img: Image.Image, in_channels: int) -> torch.Tensor:
    img_resized = pil_img.resize((224, 224))
    gray = img_resized.convert("L")
    t = transforms.ToTensor()(gray)  # [1,H,W]
    if in_channels == 1:
        out = t
    elif in_channels > 1:
        out = t.repeat(in_channels, 1, 1)
    else:
        raise BrainModelLoadError(f"Invalid brain model in_channels={in_channels}")

    # Normalize to match training typical of medical grayscale nets.
    mean = torch.full((in_channels,), 0.5, dtype=out.dtype)
    std = torch.full((in_channels,), 0.5, dtype=out.dtype)
    out = transforms.Normalize(mean=mean.tolist(), std=std.tolist())(out)
    return out

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_risk_tier(is_disease: bool, conf: float) -> str:
    if not is_disease: return "Normal"
    if conf >= 0.90: return "Critical"
    if conf >= 0.75: return "High"
    if conf >= 0.60: return "Medium"
    return "Low"

def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

# ==============================================================================
# EXPLAINABLE AI (XAI)
# ==============================================================================
def generate_chest_heatmap(pil_img: Image.Image, model: nn.Module) -> str:
    """Occlusion Sensitivity for Chest LNN."""
    img_resized = pil_img.resize((224, 224))
    tensor = chest_transforms(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        base_prob = torch.softmax(model(tensor), dim=1)[0, 1].item()

    heatmap = np.zeros((224, 224), dtype=np.float32)
    win, stride = 28, 14

    for y in range(0, 224 - win + 1, stride):
        for x in range(0, 224 - win + 1, stride):
            occ = tensor.clone()
            occ[0, 0, y:y+win, x:x+win] = 0.0
            with torch.no_grad():
                prob = torch.softmax(model(occ), dim=1)[0, 1].item()
            heatmap[y:y+win, x:x+win] += max(0.0, base_prob - prob)

    if heatmap.max() > 0: heatmap = heatmap / heatmap.max()
    import matplotlib.cm as cm
    rgba = (cm.get_cmap("jet")(heatmap) * 255).astype(np.uint8)
    heat_pil = Image.fromarray(rgba[:, :, :3]).resize(img_resized.size, Image.NEAREST)
    return pil_to_b64(Image.blend(img_resized.convert("RGB"), heat_pil, alpha=0.45))

def generate_brain_gradcam_heatmap(pil_img: Image.Image, model: nn.Module, class_idx: int, meta: Dict[str, Any]) -> str:
    """
    Minimal Grad-CAM implementation for torchvision ResNet18 (no extra deps).

    Returns a base64-encoded RGB overlay image.
    """
    img_resized = pil_img.resize((224, 224)).convert("RGB")
    input_tensor = preprocess_brain_classifier_image(img_resized, meta).unsqueeze(0).to(DEVICE)

    # Default target: last conv in the final block (matches common Grad-CAM setups).
    target_layer = getattr(getattr(model, "layer4", [None])[-1], "conv2", None)
    if target_layer is None:
        return pil_to_b64(img_resized)

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def _fwd_hook(_m, _inp, out):
        activations.append(out)

    def _bwd_hook(_m, _grad_in, grad_out):
        if grad_out and grad_out[0] is not None:
            gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()

    try:
        logits = model(input_tensor)
        score = logits[0, int(class_idx)]
        score.backward()
    finally:
        h1.remove()
        h2.remove()

    if not activations or not gradients:
        return pil_to_b64(img_resized)

    act = activations[0].detach()  # [1,C,H,W]
    grad = gradients[0].detach()   # [1,C,H,W]

    weights = grad.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))  # [1,1,H,W]
    cam = cam[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam_np = cam.cpu().numpy().astype(np.float32)
    cam_np = cv2.resize(cam_np, img_resized.size, interpolation=cv2.INTER_CUBIC)
    heatmap = (cam_np * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig = np.array(img_resized)
    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)
    return pil_to_b64(Image.fromarray(overlay))

def process_brain_unet_output(pil_img: Image.Image, output: torch.Tensor) -> Tuple[float, str]:
    """EXPERT FIX: Uses U-Net segmentation mask directly as heatmap. Ultra-Fast!"""
    img_resized = pil_img.resize((224, 224))

    if output.dim() != 4:
        raise BrainModelLoadError(f"Unexpected BrainResUNet output dims: {output.dim()} (expected 4D)")

    c = int(output.shape[1])
    if c == 1:
        mask_probs = torch.sigmoid(output)[0, 0].detach().cpu().numpy()
    else:
        # Multi-class segmentation: treat channel 0 as background and compute "any-tumor" prob.
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()  # [C,H,W]
        mask_probs = 1.0 - probs[0]

    # Calculate global confidence based on 95th percentile of the mask (removes noise)
    final_conf = float(np.percentile(mask_probs, 95))

    # Generate Heatmap image overlay directly from mask
    heatmap_norm = mask_probs / (mask_probs.max() + 1e-8)
    import matplotlib.cm as cm
    rgba = (cm.get_cmap("jet")(heatmap_norm) * 255).astype(np.uint8)
    heat_pil = Image.fromarray(rgba[:, :, :3])

    # Blend and convert
    overlay_b64 = pil_to_b64(Image.blend(img_resized.convert("RGB"), heat_pil, alpha=0.55))

    return final_conf, overlay_b64

# ==============================================================================
# ROUTES: MAIN & CHEST
# ==============================================================================
@app.route("/")
def main_home():
    return render_template("main_home.html")

@app.route("/chest")
def chest_index():
    return render_template("chest/index.html")

@app.route("/chest/upload_single", methods=["GET", "POST"])
def chest_upload_single():
    if request.method == "GET": return render_template("chest/upload_single.html")

    file = request.files.get("image")
    if not file or not allowed_file(file.filename): abort(400, "Invalid file.")

    try:
        image_bytes = file.read()
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)).convert("L"))
        tensor_img = chest_transforms(pil_img).unsqueeze(0).to(DEVICE)

        m1, m2 = load_chest_models()
        with torch.no_grad():
            p1, p2 = torch.softmax(m1(tensor_img), dim=1), torch.softmax(m2(tensor_img), dim=1)
            c1, idx1 = torch.max(p1, 1)
            c2, idx2 = torch.max(p2, 1)

        pred1, val1 = ["Normal", "Pneumonia"][idx1.item()], c1.item()
        pred2, val2 = ["Normal", "Lung Opacity"][idx2.item()], c2.item()

        f_pred, f_conf, target_m = (pred1, val1, m1) if val1 >= val2 else (pred2, val2, m2)

        orig_b64 = pil_to_b64(pil_img.convert("RGB"))
        heat_b64 = generate_chest_heatmap(pil_img, target_m)

        return render_template("chest/result_single.html",
                               image_path=orig_b64, heatmap_path=heat_b64,
                               pred1=pred1, conf1=val1, pred2=pred2, conf2=val2,
                               final_label=f_pred, final_conf=f_conf,
                               is_disease=(f_pred != "Normal"))
    except Exception as e:
        logging.error(f"Chest Inference Error: {e}")
        abort(500, "Analysis failed.")

@app.route("/chest/upload_bulk", methods=["GET", "POST"])
def chest_upload_bulk():
    if request.method == "GET": return render_template("chest/upload_bulk.html")

    files = request.files.getlist("folder")
    if not files: abort(400, "No files provided.")

    m1, m2 = load_chest_models()
    results = []

    for file in files:
        if not allowed_file(file.filename): continue
        try:
            image_bytes = file.read()
            pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)).convert("L"))
            tensor = chest_transforms(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                c1, i1 = torch.max(torch.softmax(m1(tensor), dim=1), 1)
                c2, i2 = torch.max(torch.softmax(m2(tensor), dim=1), 1)

            p1, v1 = "Pneumonia" if i1.item() == 1 else "Normal", c1.item()
            p2, v2 = "Lung Opacity" if i2.item() == 1 else "Normal", c2.item()

            f_p, f_c = (p1, v1) if v1 >= v2 else (p2, v2)
            is_dis = f_p != "Normal"

            results.append([
                secure_filename(file.filename), p1, f"{v1:.4f}", p2, f"{v2:.4f}",
                f_p, f"{f_c:.4f}", calculate_risk_tier(is_dis, f_c), time.strftime("%Y-%m-%d %H:%M:%S")
            ])
        except Exception: continue

    batch_id = uuid.uuid4().hex
    csv_path = os.path.join(REPORT_DIR, f"chest_batch_{batch_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "M1_Pred", "M1_Conf", "M2_Pred", "M2_Conf", "Final_Pred", "Final_Conf", "Risk_Tier", "Timestamp"])
        writer.writerows(sorted(results, key=lambda x: (x[5] == "Normal", -float(x[6]))))

    return redirect(url_for("chest_loading_bulk", batch_id=batch_id))

@app.route("/chest/loading/<batch_id>")
def chest_loading_bulk(batch_id):
    return render_template("chest/loading_chest.html", batch_id=batch_id)

@app.route("/chest/result_bulk/<batch_id>")
def chest_result_bulk(batch_id):
    csv_path = os.path.join(REPORT_DIR, f"chest_batch_{batch_id}.csv")
    if not os.path.exists(csv_path): abort(404, "Report expired.")
    results = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            results.append({
                "filename": row[0], "prediction1": row[1], "confidence1": float(row[2]),
                "prediction2": row[3], "confidence2": float(row[4]),
                "final_prediction": row[5], "final_confidence": float(row[6]),
                "risk_tier": row[7], "is_disease": row[5] != "Normal"
            })
    return render_template("chest/result_bulk.html", results=results, batch_id=batch_id)

# ==============================================================================
# ROUTES: BRAIN MODULE (PURE PYTORCH CLASSIFICATION + GRAD-CAM)
# ==============================================================================
@app.route("/brain")
def brain_index():
    return render_template("brain/brain_index.html")

@app.route("/brain/upload_single", methods=["GET", "POST"])
def brain_upload_single():
    if request.method == "GET": return render_template("brain/upload_brain_single.html")

    file = request.files.get("image")
    if not file or not allowed_file(file.filename): abort(400, "Invalid file.")

    try:
        image_bytes = file.read()
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        brain_model, meta = load_brain_classifier()
        tensor_img = preprocess_brain_classifier_image(pil_img, meta).unsqueeze(0).to(DEVICE)

        # CNN Classification Inference
        with torch.no_grad():
            logits = brain_model(tensor_img)
            temperature = float(meta.get("temperature", 1.0)) or 1.0
            probs = torch.softmax(logits / temperature, dim=1)[0]
            class_idx = int(torch.argmax(probs).item())

        class_names = list(meta["class_names"])
        no_tumor_idx = find_no_tumor_index(class_names)
        tumor_threshold = float(meta.get("tumor_threshold", 0.50))

        if no_tumor_idx is None:
            # Fallback to top-1 label if config doesn't specify "no tumor".
            class_name = class_names[class_idx]
            final_label = format_brain_class_label(class_name)
            final_conf = float(probs[class_idx].item())
            is_disease = is_brain_tumor_class(class_name)
            cam_class_idx = class_idx
        else:
            prob_no_tumor = float(probs[no_tumor_idx].item())
            tumor_prob = 1.0 - prob_no_tumor

            tumor_type_indices = [i for i in range(len(class_names)) if i != no_tumor_idx]
            tumor_type_idx = max(tumor_type_indices, key=lambda i: float(probs[i].item()))
            tumor_type_name = class_names[tumor_type_idx]

            is_disease = tumor_prob >= tumor_threshold
            if is_disease:
                final_label = format_brain_class_label(tumor_type_name)
                final_conf = float(tumor_prob)
                cam_class_idx = tumor_type_idx
            else:
                final_label = "Normal"
                final_conf = float(prob_no_tumor)
                cam_class_idx = no_tumor_idx

        # Explainability (Grad-CAM)
        heat_b64 = generate_brain_gradcam_heatmap(pil_img, brain_model, cam_class_idx, meta)

        # For UI/debugging.
        prob_breakdown = []
        for i, raw in enumerate(class_names):
            prob_breakdown.append({"raw": raw, "label": format_brain_class_label(raw), "prob": float(probs[i].item())})
        prob_breakdown.sort(key=lambda x: x["prob"], reverse=True)

        orig_b64 = pil_to_b64(pil_img)

        return render_template("brain/result_brain_single.html",
                               image_path=orig_b64, heatmap_path=heat_b64,
                               final_label=final_label, final_conf=final_conf,
                               is_disease=is_disease, risk_tier=calculate_risk_tier(is_disease, final_conf),
                               model_file=meta.get("model_file", ""),
                               probs=prob_breakdown)

    except BrainModelLoadError as e:
        logging.error(f"Brain Model Error: {e}", exc_info=True)
        abort(500, str(e))
    except Exception as e:
        logging.error(f"Brain Inference Error: {e}", exc_info=True)
        abort(500, "Brain processing failed.")

@app.route("/brain/upload_bulk", methods=["GET", "POST"])
def brain_upload_bulk():
    if request.method == "GET": return render_template("brain/upload_brain_bulk.html")

    files = request.files.getlist("folder")
    if not files: abort(400, "No files.")

    try:
        brain_model, meta = load_brain_classifier()
    except BrainModelLoadError as e:
        logging.error(f"Brain Model Error: {e}", exc_info=True)
        abort(500, str(e))
    results = []

    class_names = list(meta["class_names"])
    no_tumor_idx = find_no_tumor_index(class_names)
    tumor_threshold = float(meta.get("tumor_threshold", 0.50))

    for file in files:
        if not allowed_file(file.filename): continue
        try:
            image_bytes = file.read()
            pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            tensor_img = preprocess_brain_classifier_image(pil_img, meta).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = brain_model(tensor_img)
                temperature = float(meta.get("temperature", 1.0)) or 1.0
                probs = torch.softmax(logits / temperature, dim=1)[0]
                class_idx = int(torch.argmax(probs).item())

            if no_tumor_idx is None:
                class_name = class_names[class_idx]
                label = format_brain_class_label(class_name)
                final_conf = float(probs[class_idx].item())
                is_dis = is_brain_tumor_class(class_name)
            else:
                prob_no_tumor = float(probs[no_tumor_idx].item())
                tumor_prob = 1.0 - prob_no_tumor

                tumor_type_indices = [i for i in range(len(class_names)) if i != no_tumor_idx]
                tumor_type_idx = max(tumor_type_indices, key=lambda i: float(probs[i].item()))
                tumor_type_name = class_names[tumor_type_idx]

                is_dis = tumor_prob >= tumor_threshold
                if is_dis:
                    label = format_brain_class_label(tumor_type_name)
                    final_conf = float(tumor_prob)
                else:
                    label = "Normal"
                    final_conf = float(prob_no_tumor)

            results.append([
                secure_filename(file.filename), label, f"{final_conf:.4f}",
                calculate_risk_tier(is_dis, final_conf), time.strftime("%Y-%m-%d %H:%M:%S")
            ])
        except Exception: continue

    batch_id = uuid.uuid4().hex
    csv_path = os.path.join(REPORT_DIR, f"brain_batch_{batch_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Final_Prediction", "Final_Confidence", "Risk_Tier", "Timestamp"])
        writer.writerows(sorted(results, key=lambda x: (x[1] == "Normal", -float(x[2]))))

    return redirect(url_for("brain_loading_bulk", batch_id=batch_id))

@app.route("/brain/loading/<batch_id>")
def brain_loading_bulk(batch_id):
    return render_template("brain/loading_brain.html", batch_id=batch_id)

@app.route("/brain/result_bulk/<batch_id>")
def brain_result_bulk(batch_id):
    csv_path = os.path.join(REPORT_DIR, f"brain_batch_{batch_id}.csv")
    if not os.path.exists(csv_path): abort(404, "Report expired.")

    results = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            results.append({
                "filename": row[0], "final_prediction": row[1],
                "final_confidence": float(row[2]), "risk_tier": row[3],
                "is_disease": row[1] != "Normal"
            })
    return render_template("brain/result_brain_bulk.html", results=results, batch_id=batch_id)

# ==============================================================================
# DOWNLOADS & ERROR HANDLERS
# ==============================================================================
@app.route("/reports/download/<module>/<batch_id>")
def download_report(module, batch_id):
    csv_path = os.path.join(REPORT_DIR, f"{module}_batch_{batch_id}.csv")
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name=f"{module.capitalize()}_Batch_Report.csv")
    abort(404, "File expired.")

def wants_json_response() -> bool:
    if request.is_json:
        return True
    accept = request.headers.get("Accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return True
    return False

@app.errorhandler(400)
def bad_req(e):
    msg = getattr(e, "description", str(e))
    if wants_json_response():
        return jsonify(error=msg), 400
    return render_template("error.html", code=400, title="Bad Request", message=msg), 400

@app.errorhandler(404)
def not_fnd(e):
    msg = getattr(e, "description", "Resource expired or not found. Please upload again.")
    if wants_json_response():
        return jsonify(error=msg), 404
    return render_template("error.html", code=404, title="Not Found", message=msg), 404

@app.errorhandler(413)
def too_large(e):
    msg = getattr(e, "description", "Batch limit 512MB exceeded.")
    if wants_json_response():
        return jsonify(error=msg), 413
    return render_template("error.html", code=413, title="Upload Too Large", message=msg), 413

@app.errorhandler(500)
def internal(e):
    msg = getattr(e, "description", "Server Processing Error.")
    if wants_json_response():
        return jsonify(error=msg), 500
    return render_template("error.html", code=500, title="Server Error", message=msg), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
