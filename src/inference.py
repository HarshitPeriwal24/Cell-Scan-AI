
import ctypes
import gc
import os

import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

from model import load_model
from segmentation import extract_cells, is_single_cell_crop

# Small cloud instances get a fraction of one CPU, but torch spawns one
# thread per *host* core; the threads then fight over the quota and
# inference becomes many times slower. One thread is fastest there.
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

# Load model once
model, device, config = load_model()

# A cell counts as infected when P(Parasitized) reaches this value.
CELL_THRESHOLD = float(os.getenv("CELL_THRESHOLD", "0.5"))
MAX_CELLS = int(os.getenv("MAX_CELLS", "120"))

CLASS_NAMES = config["class_names"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def create_gradcam(model, input_tensor, target_class):
    """
    Generate Grad-CAM using native PyTorch hooks.
    No pytorch-grad-cam or scikit-learn required.
    """

    target_layer = model.layer4[-1]

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.update({"value": grad})
        )

    hook = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        # Forward pass with gradients enabled
        with torch.enable_grad():
            outputs = model(input_tensor)

            score = outputs[:, target_class].sum()
            score.backward()

        activation = activations["value"].detach()
        gradient = gradients["value"].detach()

        # Average gradients over spatial dimensions
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * activation, dim=1)

        # Remove negative activations
        cam = F.relu(cam)

        # Resize heatmap to input image dimensions
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        # Normalize heatmap
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def create_visualization(original_image, grayscale_cam):
    """
    Create a Grad-CAM overlay using NumPy and Pillow.
    """

    # Convert grayscale heatmap to RGB JET-like colors
    x = np.clip(grayscale_cam, 0, 1)

    red = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)

    heatmap = np.stack(
        [red, green, blue],
        axis=-1
    )

    original = np.asarray(original_image).astype(np.float32) / 255.0

    # Overlay heatmap on original image
    overlay = (
        0.5 * original +
        0.5 * heatmap
    )

    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def _release_memory():
    """Return freed memory to the OS (keeps the 512 MB free tier alive)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # not glibc (Windows / macOS dev machines)


def _classify(crops, batch_size=8):
    """Return P(Parasitized) for each PIL crop."""
    par_idx = CLASS_NAMES.index("Parasitized")
    probs = []
    with torch.inference_mode():
        for i in range(0, len(crops), batch_size):
            batch = torch.stack([transform(c) for c in crops[i:i + batch_size]]).to(device)
            p = torch.softmax(model(batch), dim=1)[:, par_idx]
            probs.extend(p.cpu().tolist())
    return probs


def _gradcam_panel(cell_image, target_class, output_path, left_title="Original Image"):
    """Save the side-by-side original / Grad-CAM panel for one cell."""
    cell_image = cell_image.convert("RGB").resize((224, 224))
    input_tensor = transform(cell_image).unsqueeze(0).to(device)
    grayscale_cam = create_gradcam(model, input_tensor, target_class)
    gradcam_image = Image.fromarray(create_visualization(cell_image, grayscale_cam))

    combined = Image.new("RGB", (448, 244), color=(25, 25, 25))
    combined.paste(cell_image, (0, 20))
    combined.paste(gradcam_image, (224, 20))
    draw = ImageDraw.Draw(combined)
    draw.text((10, 3), left_title, fill="white")
    draw.text((234, 3), "Grad-CAM", fill="white")
    combined.save(output_path)


def _single_cell_result(image, stem):
    p_par = _classify([image])[0]
    probs = {"Parasitized": p_par, "Uninfected": 1 - p_par}
    prediction = max(probs, key=probs.get)

    gradcam_path = OUTPUT_DIR / f"{stem}_gradcam.png"
    _gradcam_panel(image, CLASS_NAMES.index(prediction), gradcam_path)

    return {
        "mode": "cell",
        "prediction": prediction,
        "confidence": round(probs[prediction] * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in probs.items()},
        "gradcam_path": str(gradcam_path),
    }


def _slide_result(image, cells, stem):
    crops = [c["crop"] for c in cells]
    probs = _classify(crops)
    infected = [i for i, p in enumerate(probs) if p >= CELL_THRESHOLD]

    # Annotated field: red = parasitized, green = uninfected
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = max(2, round(max(image.size) / 400))
    for i, cell in enumerate(cells):
        colour = (230, 40, 40) if i in infected else (40, 190, 90)
        draw.rectangle(cell["box"], outline=colour, width=width)
    annotated_path = OUTPUT_DIR / f"{stem}_annotated.jpg"
    annotated.save(annotated_path, quality=88)

    # Grad-CAM for the most suspicious cell
    top = int(np.argmax(probs))
    top_class = "Parasitized" if probs[top] >= CELL_THRESHOLD else "Uninfected"
    gradcam_path = OUTPUT_DIR / f"{stem}_gradcam.png"
    _gradcam_panel(crops[top], CLASS_NAMES.index(top_class), gradcam_path,
                   left_title="Most suspicious cell")

    prediction = "Parasitized" if infected else "Uninfected"
    # Slide-level score: how strongly the most suspicious cell is infected
    p_slide = probs[top]
    confidence = p_slide if infected else 1 - p_slide

    return {
        "mode": "slide",
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "Parasitized": round(p_slide * 100, 2),
            "Uninfected": round((1 - p_slide) * 100, 2),
        },
        "cells_analyzed": len(cells),
        "cells_parasitized": len(infected),
        "parasitemia": round(100 * len(infected) / len(cells), 2),
        "gradcam_path": str(gradcam_path),
        "annotated_path": str(annotated_path),
    }


def analyze_image(image_path):
    """
    Classify an uploaded image.

    - NIH-style single-cell crop  -> classify it directly (mode "cell")
    - microscope field / photo    -> segment every red blood cell, classify
                                     each one, report infected-cell count
                                     and parasitemia (mode "slide")
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")   # phone photos
    stem = image_path.stem

    try:
        if is_single_cell_crop(image):
            return _single_cell_result(image, stem)

        cells, _ = extract_cells(image, max_cells=MAX_CELLS)
        if not cells:
            raise ValueError(
                "No blood cells were found in this image. Upload a microscope "
                "image of a stained thin blood smear, or a single cropped cell."
            )
        return _slide_result(image, cells, stem)
    finally:
        _release_memory()



if __name__ == "__main__":

    image_path = input("Enter blood cell image path: ")

    result = analyze_image(image_path)

    print("\nCell Scan AI - Analysis Results")
    print("--------------------------------")

    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")
    print("Probabilities:", result["probabilities"])
    print("Grad-CAM saved:", result["gradcam_path"])
