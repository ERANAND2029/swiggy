import os
import io
import time
import base64
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI(title="Indian Food Live Detector", version="1.0.0")

# Enable CORS for local testing and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Colors palette for the classes
CLASS_COLORS = {
    "chillipotato": "#ef4444",  # Red
    "dosa": "#f59e0b",          # Amber
    "kaju katli": "#ec4899",    # Pink
    "ladu": "#eab308",          # Yellow
    "momos": "#06b6d4",         # Cyan
    "panner tikka": "#f97316",   # Orange
    "samosa": "#10b981",        # Emerald Green
    "vada pav": "#8b5cf6",      # Purple
}

# Resolve model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model", "detectanand.pt"))

print(f"Loading YOLO Model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names if hasattr(model, "names") else {}
print(f"Model loaded successfully with classes: {CLASS_NAMES}")

# Serve static assets
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Base64DetectRequest(BaseModel):
    image: str
    conf: Optional[float] = 0.25
    iou: Optional[float] = 0.45
    return_annotated: Optional[bool] = False


def run_yolo_detection(image: Image.Image, conf: float = 0.25, iou: float = 0.45, return_annotated: bool = True):
    start_time = time.time()
    
    # Run Ultralytics inference
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    infer_ms = round((time.time() - start_time) * 1000, 1)

    result = results[0]
    detections = []
    count_by_class = {}

    orig_w, orig_h = image.size

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes_xyxy, confs, classes):
            cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            x1, y1, x2, y2 = [round(float(coord), 1) for coord in box]

            # Normalize coordinates (0-1) for flexible client-side rendering
            norm_box = [
                round(x1 / orig_w, 4),
                round(y1 / orig_h, 4),
                round(x2 / orig_w, 4),
                round(y2 / orig_h, 4),
            ]

            color = CLASS_COLORS.get(cls_name.lower(), "#3b82f6")
            detections.append({
                "class_id": int(cls_id),
                "class_name": cls_name,
                "confidence": round(float(score), 3),
                "box": [x1, y1, x2, y2],
                "box_norm": norm_box,
                "color": color,
            })
            count_by_class[cls_name] = count_by_class.get(cls_name, 0) + 1

    annotated_base64 = None
    if return_annotated:
        # result.plot() returns BGR numpy array
        plotted_bgr = result.plot()
        # Convert BGR to RGB
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
        pil_res = Image.fromarray(plotted_rgb)
        
        buffered = io.BytesIO()
        pil_res.save(buffered, format="JPEG", quality=85)
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "success": True,
        "width": orig_w,
        "height": orig_h,
        "inference_ms": infer_ms,
        "detections": detections,
        "count_by_class": count_by_class,
        "annotated_image": annotated_base64,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h3>Live Food Detector Frontend is building...</h3>")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Indian Food Live Detector",
        "classes_count": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    }


@app.get("/api/info")
async def get_model_info():
    return {
        "model": os.path.basename(MODEL_PATH),
        "classes": CLASS_NAMES,
        "colors": CLASS_COLORS,
    }


@app.post("/api/predict-file")
async def predict_from_file(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    return_annotated: bool = Form(True)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        res = run_yolo_detection(image, conf=conf, iou=iou, return_annotated=return_annotated)
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


@app.post("/api/predict-frame")
async def predict_from_frame(payload: Base64DetectRequest):
    try:
        img_str = payload.image
        if "," in img_str:
            img_str = img_str.split(",")[1]
        decoded = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(decoded)).convert("RGB")
        res = run_yolo_detection(
            image,
            conf=payload.conf or 0.25,
            iou=payload.iou or 0.45,
            return_annotated=payload.return_annotated or False
        )
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
