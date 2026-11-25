
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms

from model import load_model


# Load model once
model, device, config = load_model()

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


def analyze_image(image_path):
    """
    Predict malaria cell class and generate Grad-CAM.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load original image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    img_np = np.array(image).astype(np.float32) / 255.0

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    pred_class = outputs.argmax(dim=1).item()

    prediction = CLASS_NAMES[pred_class]
    confidence = probabilities[pred_class].item() * 100

    # Generate Grad-CAM
    grayscale_cam = create_gradcam(
        model,
        input_tensor,
        pred_class
    )

    visualization = create_visualization(
        image,
        grayscale_cam
    )

    # Save side-by-side visualization
    output_path = OUTPUT_DIR / f"{image_path.stem}_gradcam.png"

    original_image = image.copy()
    gradcam_image = Image.fromarray(visualization)

    combined_image = Image.new(
        "RGB",
        (448, 244),
        color=(25, 25, 25)
    )

    combined_image.paste(original_image, (0, 20))
    combined_image.paste(gradcam_image, (224, 20))

    draw = ImageDraw.Draw(combined_image)

    draw.text((10, 3), "Original Image", fill="white")
    draw.text((234, 3), "Grad-CAM", fill="white")

    combined_image.save(output_path)

    # Return complete results
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(
                probabilities[i].item() * 100, 2
            )
            for i in range(len(CLASS_NAMES))
        },
        "gradcam_path": str(output_path)
    }


if __name__ == "__main__":

    image_path = input("Enter blood cell image path: ")

    result = analyze_image(image_path)

    print("\nCell Scan AI - Analysis Results")
    print("--------------------------------")

    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")
    print("Probabilities:", result["probabilities"])
    print("Grad-CAM saved:", result["gradcam_path"])
