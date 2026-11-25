
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import load_model


# Load trained model
model, device, config = load_model()

CLASS_NAMES = config["class_names"]

# Project paths
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


def generate_gradcam(image_path):
    """
    Generate Grad-CAM visualization for a blood cell image.
    """

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    # Convert image to NumPy for visualization
    img_np = np.array(image).astype(np.float32) / 255.0

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    # Predict class
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    pred_class = outputs.argmax(dim=1).item()

    # Grad-CAM target layer
    target_layer = model.layer4[-1]

    # Generate Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    targets = [ClassifierOutputTarget(pred_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Overlay heatmap
    visualization = show_cam_on_image(
        img_np,
        grayscale_cam,
        use_rgb=True
    )

    # Save visualization
    output_path = OUTPUT_DIR / f"{Path(image_path).stem}_gradcam.png"

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    cam.activations_and_grads.release()

    return {
        "prediction": CLASS_NAMES[pred_class],
        "confidence": round(
            probabilities[pred_class].item() * 100, 2
        ),
        "gradcam_path": str(output_path)
    }


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    result = generate_gradcam(image_path)

    print("\nGrad-CAM Results")
    print("----------------------")
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")
    print("Saved visualization:", result["gradcam_path"])
