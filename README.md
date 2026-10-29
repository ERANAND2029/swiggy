# 🍲 Indian Food Live Vision Detector (Swiggy Food Vision)

An ultra-lightweight, high-speed live web application for real-time Indian Food Object Detection using **FastAPI**, **YOLOv8**, and **HTML5 Canvas**. Optimized specifically to deploy seamlessly on **Render (Free Tier)**.

---

## 🍕 Supported Food Classes (8 Classes)
- 🌶️ **Chilli Potato**
- 🥞 **Dosa**
- 💎 **Kaju Katli**
- 🟡 **Laddoo**
- 🥟 **Momos**
- 🍢 **Paneer Tikka**
- 🔺 **Samosa**
- 🍔 **Vada Pav**

---

## ✨ Features
1. **Live Camera Mode**:
   - Real-time webcam & mobile back camera support (`getUserMedia`).
   - HTML5 Canvas overlay for real-time bounding boxes and labels without lag.
   - Flip camera switch (Front / Back camera for mobile devices).
   - Adjustable confidence threshold and frame interval slider.
   - Live item count badges and snapshot capture.
2. **Photo Upload Mode**:
   - Drag & drop or browse food images.
   - Instant YOLO inference with annotated bounding box image.
   - Breakdown count of detected food items.
   - One-click annotated image download.
3. **Render Free-Tier Optimized**:
   - Uses CPU-optimized PyTorch wheels (~200MB memory footprint, avoids 512MB RAM OOM).
   - Async FastAPI backend with sub-second response times.
   - Built-in Dockerfile & `render.yaml` for 1-click deploy.

---

## 🚀 Local Run (PC Testing)

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python main.py
   ```
   Or:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

3. **Open in Browser**:
   ```
   http://localhost:8000
   ```

---

## 🌐 Deploy to Render (Step-by-Step)

### Option A: Via Docker (Recommended - Prevents any dependency mismatch)
1. Go to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository: `ERANAND2029/swiggy`.
4. Choose **Environment**: **Docker**.
5. Choose **Instance Type**: **Free**.
6. Click **Deploy Web Service**!

### Option B: Via Python Native
1. Go to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect GitHub repository: `ERANAND2029/swiggy`.
4. Set:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy Web Service**!
