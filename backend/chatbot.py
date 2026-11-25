
# CellScan AI - Enhanced Rule-Based Knowledge Assistant

def get_chatbot_response(message: str) -> str:

    message = message.lower().strip()

    if not message:
        return "Please enter a question."

    # ---------------- GREETINGS ----------------
    if any(word in message for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        return (
            "Hello! Welcome to CellScan AI.\n\n"
            "I'm your CellScan Assistant. I can explain:\n"
            "1. ResNet18 architecture\n"
            "2. Grad-CAM visualization\n"
            "3. NIH Malaria dataset\n"
            "4. Model training and predictions\n"
            "5. Model limitations\n\n"
            "What would you like to know?"
        )

    # ---------------- GRAD-CAM ----------------
    if any(word in message for word in [
        "gradcam", "grad-cam", "heatmap",
        "visualization", "explainability", "explainable ai"
    ]):
        return (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) "
            "is an Explainable AI technique used to visualize which "
            "regions of an image influenced a CNN's prediction.\n\n"

            "How Grad-CAM works in CellScan AI:\n"
            "1. The input image is passed through ResNet18.\n"
            "2. The model generates a classification prediction.\n"
            "3. Gradients are calculated for the predicted class.\n"
            "4. These gradients are used to generate an activation heatmap.\n"
            "5. The heatmap is overlaid on the original microscopic image.\n\n"

            "Interpretation:\n"
            "Red and yellow regions generally represent stronger "
            "activation, while cooler regions represent weaker activation.\n\n"

            "Why do we use Grad-CAM?\n"
            "It makes our model more interpretable by showing which "
            "image regions contributed to its prediction. However, "
            "it does not prove that the model's reasoning is medically correct."
        )

    # ---------------- RESNET18 / MODEL ARCHITECTURE ----------------
    if any(word in message for word in [
        "resnet", "architecture", "cnn",
        "neural network", "skip connection",
        "residual connection", "transfer learning",
        "deep learning", "layers", "model"
    ]):
        return (
            "CellScan AI uses ResNet18 (Residual Network 18), "
            "a Convolutional Neural Network (CNN) architecture "
            "designed for image classification.\n\n"

            "What is ResNet18?\n"
            "It contains 18 weighted layers and uses residual or "
            "skip connections. These connections help information "
            "and gradients flow through deeper neural networks.\n\n"

            "How we use it in CellScan AI:\n"
            "1. We start with a pretrained ResNet18 model.\n"
            "2. The pretrained network is used as the feature extractor.\n"
            "3. Its final fully connected layer is replaced with a "
            "custom classification layer.\n"
            "4. The new layer produces two output classes:\n"
            "   - Parasitized\n"
            "   - Uninfected\n\n"

            "Why ResNet18?\n"
            "It offers a useful balance between classification "
            "performance and computational efficiency, making it "
            "suitable for our laptop-based project.\n\n"

            "We use transfer learning to benefit from features "
            "learned during ImageNet pretraining."
        )

    # ---------------- DATASET ----------------
    if any(word in message for word in [
        "dataset", "data", "images",
        "nih", "training data", "samples"
    ]):
        return (
            "CellScan AI uses the NIH Malaria Cell Images Dataset.\n\n"

            "Dataset details:\n"
            "- Total images: 27,558\n"
            "- Parasitized images: 13,779\n"
            "- Uninfected images: 13,779\n"
            "- Number of classes: 2\n"
            "- Image type: Microscopic blood-cell images\n\n"

            "The dataset contains two categories:\n\n"

            "1. Parasitized:\n"
            "Images containing malaria-infected blood cells.\n\n"

            "2. Uninfected:\n"
            "Images containing uninfected blood cells.\n\n"

            "The dataset is used to train and evaluate our "
            "ResNet18 image-classification model."
        )

    # ---------------- TRAINING ----------------
    if any(word in message for word in [
        "training", "train", "optimizer",
        "loss", "epochs", "preprocessing",
        "augmentation", "learning rate"
    ]):
        return (
            "CellScan AI uses PyTorch to train its ResNet18 model.\n\n"

            "Our training pipeline includes:\n"
            "1. Loading and preprocessing microscopic cell images.\n"
            "2. Resizing images to 224 x 224 pixels.\n"
            "3. Normalizing images using ImageNet statistics.\n"
            "4. Applying data augmentation during training.\n"
            "5. Using the Adam optimizer.\n"
            "6. Using CrossEntropyLoss for classification.\n"
            "7. Selecting the best checkpoint based on validation performance.\n\n"

            "The model learns to distinguish between Parasitized "
            "and Uninfected blood-cell images."
        )

    # ---------------- PREDICTION ----------------
    if any(word in message for word in [
        "prediction", "predict", "confidence",
        "upload", "how to use", "how does it work"
    ]):
        return (
            "You can use CellScan AI through its Live Demo section.\n\n"

            "Steps:\n"
            "1. Open the Live Demo section.\n"
            "2. Upload a blood-cell microscopy image.\n"
            "3. Click the Detect Malaria button.\n"
            "4. The backend processes the image using ResNet18.\n"
            "5. The system displays the predicted class and confidence.\n"
            "6. A Grad-CAM visualization is also displayed when available.\n\n"

            "The two possible predictions are Parasitized and Uninfected.\n\n"

            "Note: Model confidence is not the same as overall "
            "model accuracy."
        )

    # ---------------- ACCURACY AND RELIABILITY ----------------
    if any(word in message for word in [
        "accuracy", "accurate", "reliable",
        "performance", "test score", "metrics"
    ]):
        return (
            "Model accuracy measures the proportion of correctly "
            "classified samples in an evaluation dataset.\n\n"

            "Confidence is different: it represents the model's "
            "estimated probability for a particular prediction.\n\n"

            "For CellScan AI, actual accuracy should be reported "
            "using the recorded test-set evaluation results. "
            "A high confidence score on one image does not establish "
            "overall model accuracy.\n\n"

            "Other useful evaluation metrics include precision, "
            "recall, F1-score and a confusion matrix."
        )

    # ---------------- LIMITATIONS ----------------
    if any(word in message for word in [
        "limitation", "limitations", "disadvantage",
        "weakness", "problem", "medical diagnosis",
        "diagnosis", "clinical"
    ]):
        return (
            "CellScan AI has several important limitations:\n\n"

            "1. Dataset dependency:\n"
            "Model performance depends on the quality and diversity "
            "of the training dataset.\n\n"

            "2. Image quality:\n"
            "Blurry, poorly illuminated or unusual images may affect predictions.\n\n"

            "3. Generalization:\n"
            "Performance on images from different laboratories or "
            "microscopes may vary.\n\n"

            "4. Explainability:\n"
            "Grad-CAM highlights influential image regions but does "
            "not prove that the model's reasoning is medically correct.\n\n"

            "5. Medical limitations:\n"
            "CellScan AI is an educational and research prototype. "
            "It is not a medical diagnostic tool and cannot replace "
            "professional laboratory testing or medical advice."
        )

    # ---------------- DEFAULT RESPONSE ----------------
    return (
        "I'm CellScan Assistant, your AI project guide.\n\n"
        "I can answer questions about:\n"
        "- ResNet18 architecture and transfer learning\n"
        "- Grad-CAM and Explainable AI\n"
        "- NIH Malaria Cell Images Dataset\n"
        "- Model training and preprocessing\n"
        "- Predictions and confidence scores\n"
        "- Accuracy and evaluation metrics\n"
        "- Model limitations\n\n"
        "Please ask me something about CellScan AI!"
    )