import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet18

# ImageFolder order from the training notebook. Do not sort or rename.
CLASSES = [
    ("AnnualCrop", "Annual crop"),
    ("Forest", "Forest"),
    ("HerbaceousVegetation", "Herbaceous vegetation"),
    ("Highway", "Highway"),
    ("Industrial", "Industrial"),
    ("Pasture", "Pasture"),
    ("PermanentCrop", "Permanent crop"),
    ("Residential", "Residential"),
    ("River", "River"),
    ("SeaLake", "Sea or lake"),
]

MODEL_PATH = Path(__file__).resolve().parent / "model" / "best_model.pth"
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}")

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="Satellite land classification", lifespan=lifespan)

# Comma-separated list. "*" is ignored so an unset or leftover wildcard cannot open the API.
_DEFAULT_ORIGINS = ",".join(
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
)
_origins = os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).strip()
if not _origins or _origins == "*":
    _origins = _DEFAULT_ORIGINS
allow_origins = [item.strip().rstrip("/") for item in _origins.split(",") if item.strip() and item.strip() != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@torch.inference_mode()
def classify(model, image: Image.Image):
    tensor = TRANSFORM(image).unsqueeze(0)
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores = [
        {
            "label": label,
            "name": name,
            "confidence": round(float(probabilities[index]), 6),
        }
        for index, (label, name) in enumerate(CLASSES)
    ]
    scores.sort(key=lambda item: item["confidence"], reverse=True)
    best = scores[0]
    return {
        "label": best["label"],
        "name": best["name"],
        "confidence": best["confidence"],
        "scores": scores,
    }


def read_image(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail="Upload an image file.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image must be 8 MB or smaller.")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="That file is not a readable image.") from error
    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(status_code=400, detail="Image must be under 20 megapixels.")
    return image


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "model": "resnet18", "classes": len(CLASSES)}


@app.get("/classes")
def classes():
    return [{"label": label, "name": name} for label, name in CLASSES]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    image = read_image(raw)
    return await asyncio.to_thread(classify, app.state.model, image)
