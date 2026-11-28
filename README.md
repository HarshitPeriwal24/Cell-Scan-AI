# CellScan AI

Malaria detection in Giemsa-stained thin blood smears using a fine-tuned
ResNet18 (PyTorch), Grad-CAM explanations, a FastAPI backend and a React + Vite
frontend, with a Gemini-powered assistant that answers questions about the project.

**Live site:** https://cell-scan-ai.vercel.app

> Educational prototype. Not a medical device and not a substitute for
> professional diagnosis.

## How it works

The classifier was trained on the
[NIH Malaria Cell Images dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
(27,558 single-cell crops, 50/50 Parasitized/Uninfected). Held-out test accuracy: **96.8%**.

The API accepts two kinds of images:

| Upload | What happens |
|---|---|
| A single cropped cell (NIH style, black background) | Classified directly |
| A full microscope field with many cells | Each red blood cell is segmented ([src/segmentation.py](src/segmentation.py)), cropped NIH-style and classified; the response gives infected-cell count, parasitemia, an annotated image and Grad-CAM of the most suspicious cell |

Slide verdicts are **Parasitized** (>=2 cells with P >= 0.90, or one >= 0.97),
**Suspicious** (some cell >= 0.5 but weaker evidence; the per-cell model flags
~2-3% of healthy cells) or **Uninfected**. Thresholds can be changed with the
`CELL_THRESHOLD`, `STRONG_CELL` and `DEFINITE_CELL` environment variables.

Images where cells are too small (low magnification) are rejected; use a 100x
oil-immersion field.

## Project layout

```
backend/     FastAPI app (app.py) and Gemini assistant (llm.py)
src/         model loading, segmentation, inference + Grad-CAM
models/      trained weights (best_resnet18_malaria.pth) and config
notebooks/   training notebook
frontend/    React + Vite site
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
echo GEMINI_API_KEY=your-key > .env
uvicorn backend.app:app --reload  # http://127.0.0.1:8000

cd frontend
npm install
npm run dev                       # http://localhost:5173
```

The dataset is not in the repo; download it into `cell_images/cell_images/` to
retrain with the notebook.

## Deployment

- **Backend (Render):** build `pip install -r requirements.txt`, start
  `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`, env var `GEMINI_API_KEY`.
  `requirements.txt` is deliberately lean (CPU-only torch) so it fits the free
  512 MB instance. `GET /api/health` reports memory use.
- **Frontend (Vercel):** set `VITE_API_BASE_URL` to the Render URL.
