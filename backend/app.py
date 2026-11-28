import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from backend.llm import generate_response

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Allow backend to import your existing ML modules
sys.path.insert(0, str(SRC_DIR))

from inference import analyze_image

# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="CellScan AI API",
    description="Malaria detection using ResNet18 and Grad-CAM",
    version="1.0.0"
)

# Frontend development and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://cell-scan-ai.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve generated Grad-CAM images
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to CellScan AI API",
        "status": "running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "project": "CellScan AI"
    }

# --------------------------------------------------
# IMAGE PREDICTION
# --------------------------------------------------

# --------------------------------------------------
# CELL SCAN AI CHATBOT
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(default_factory=list)  

@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = generate_response(
        request.message,
        request.history
    )
    return {"success": True, "reply": response}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = {
        ".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"
    }

    temp_dir = PROJECT_ROOT / "outputs" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, WebP, BMP or TIFF image."
        )

    image_path = temp_dir / f"{uuid.uuid4().hex}{extension}"

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must be below 10 MB."
            )

        image_path.write_bytes(contents)

        try:
            Image.open(image_path).verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a readable image."
            )

        try:
            result = analyze_image(image_path)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        response = {
            "success": True,
            "mode": result["mode"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "gradcam_url":
                f"/outputs/gradcam/{Path(result['gradcam_path']).name}"
        }

        if result["mode"] == "slide":
            response.update({
                "cells_analyzed": result["cells_analyzed"],
                "cells_parasitized": result["cells_parasitized"],
                "parasitemia": result["parasitemia"],
                "annotated_url":
                    f"/outputs/gradcam/{Path(result['annotated_path']).name}"
            })

        return response

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    finally:
        await file.close()
        if image_path.exists():
            image_path.unlink()
