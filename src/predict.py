
import torch
from torchvision import transforms
from PIL import Image

from model import load_model

# Load trained model
model, device, config = load_model()

CLASS_NAMES = config["class_names"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    """
    Predict whether a blood cell image is
    Parasitized or Uninfected.
    """

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }

if __name__ == "__main__":
    image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print("\nPrediction Results")
    print("----------------------")
    print("Class:", result["prediction"])
    print("Confidence:", result["confidence"], "%")
    print("Probabilities:", result["probabilities"])