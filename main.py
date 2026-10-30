import os
import io
import time
import base64
import gc
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2

# PyTorch low-memory and CPU thread optimization for Render Free Tier (512MB RAM)
import torch
torch.set_grad_enabled(False)
# Limit CPU threads to prevent CPU spikes and excessive memory pooling on Render
try:
    torch.set_num_threads(2)
except Exception:
    pass

# Ensure PyTorch 2.6+ unpickler allows Ultralytics DetectionModel
try:
    import ultralytics.nn.tasks
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            ultralytics.nn.tasks.DetectionModel,
            ultralytics.nn.tasks.BaseModel,
        ])
except Exception as e:
    print(f"Safe globals warning (non-fatal): {e}")

from ultralytics import YOLO

app = FastAPI(title="Indian Food Live Detector (YOLO 26l)", version="2.0.0")

# Enable CORS for cross-origin clients and mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Colors palette for the 8 Indian food classes
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

# Lazy load model with error capture so server starts reliably
model = None
model_load_error = None
CLASS_NAMES = {
    0: 'chillipotato', 1: 'dosa', 2: 'kaju katli', 3: 'ladu',
    4: 'momos', 5: 'panner tikka', 6: 'samosa', 7: 'vada pav'
}


def load_model_instance():
    global model, model_load_error, CLASS_NAMES
    if model is not None:
        return model
    try:
        print(f"Loading YOLO26l Model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        loaded = YOLO(MODEL_PATH)
        if hasattr(loaded, "names") and loaded.names:
            CLASS_NAMES = loaded.names
        
        # Warmup model once with small input to prevent first-request lag
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        loaded.predict(dummy, imgsz=320, verbose=False)
        
        model = loaded
        print(f"YOLO26l loaded successfully! Detected classes: {CLASS_NAMES}")
        return model
    except Exception as e:
        model_load_error = str(e)
        print(f"CRITICAL: Failed to load YOLO26l model: {e}")
        return None


# Preload model at startup
@app.on_event("startup")
def startup_event():
    load_model_instance()


# Serve static assets
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Base64DetectRequest(BaseModel):
    image: str
    conf: Optional[float] = 0.25
    iou: Optional[float] = 0.45
    return_annotated: Optional[bool] = False
    imgsz: Optional[int] = 480


def run_yolo_detection(image: Image.Image, conf: float = 0.25, iou: float = 0.45, return_annotated: bool = True, imgsz: int = 480):
    m = load_model_instance()
    if m is None:
        raise RuntimeError(f"YOLO26l model failed to load on server: {model_load_error}")

    start_time = time.time()
    
    # Run inference with memory efficiency
    with torch.inference_mode():
        results = m.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        
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

            # Normalize coordinates (0-1) for client canvas rendering
            norm_box = [
                round(max(0.0, min(1.0, x1 / orig_w)), 4),
                round(max(0.0, min(1.0, y1 / orig_h)), 4),
                round(max(0.0, min(1.0, x2 / orig_w)), 4),
                round(max(0.0, min(1.0, y2 / orig_h)), 4),
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
        plotted_bgr = result.plot()
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
        pil_res = Image.fromarray(plotted_rgb)
        
        buffered = io.BytesIO()
        pil_res.save(buffered, format="JPEG", quality=80)
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
    m = load_model_instance()
    return {
        "status": "ok" if m is not None else "model_error",
        "service": "Indian Food Live Detector (YOLO 26l)",
        "model_loaded": m is not None,
        "model_error": model_load_error,
        "classes_count": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    }


@app.get("/api/info")
async def get_model_info():
    m = load_model_instance()
    return {
        "model": os.path.basename(MODEL_PATH),
        "architecture": "YOLO26l",
        "loaded": m is not None,
        "error": model_load_error,
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
        res = run_yolo_detection(image, conf=conf, iou=iou, return_annotated=return_annotated, imgsz=640)
        return JSONResponse(res)
    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


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
            return_annotated=payload.return_annotated or False,
            imgsz=payload.imgsz or 480
        )
        return JSONResponse(res)
    except Exception as e:
        print(f"Frame error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    # Render passes port via $PORT
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
