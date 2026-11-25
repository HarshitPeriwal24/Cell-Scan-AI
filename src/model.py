
import torch
import torch.nn as nn
from torchvision import models
import json
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

CONFIG_PATH = MODEL_DIR / "model_config.json"
CHECKPOINT_PATH = MODEL_DIR / "best_resnet18_malaria.pth"


def load_config():
    """Load model configuration."""
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


def load_model():
    """Load trained ResNet18 model."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = load_config()

    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(
        num_features,
        config["num_classes"]
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print("Device:", device)
    print("Classes:", config["class_names"])

    return model, device, config
