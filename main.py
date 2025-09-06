from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

# ---------- FastAPI Setup ----------
app = FastAPI()

allow_origins = ["*"]

# Allow frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("forged_unforged_educational.pt")  # Replace with your trained model path


# ---------- Response Model ----------
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2]

@app.get("/")
def root():
    return {"message": "YOLO API is running 🚀"}

# ---------- API Endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image from request
    contents = await file.read()

    # Convert to RGB (fixes 4-channel issue)
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # Run inference
    results = model(img)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    return {"detections": detections}
