# Camate Live AI Vision & Multi-Stream PTZ Controller

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8%20GPU-00FFFF?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A real-time computer vision, object detection, and ONVIF PTZ camera controller suite for IP surveillance cameras (Camate, RTSP, and ONVIF compliant hardware) featuring desktop GUI, web-based dashboard, and multi-camera stream grids.

---

## 🌟 Key Features

- **Real-Time YOLOv8 Object Detection**: High-performance GPU inference for detecting people, vehicles, backpacks, and everyday items with bounding boxes and confidence score badges.
- **OpenCV Face Detection**: Fast, lightweight Haar-cascade facial landmark bounding box tracking.
- **Hardware Pan-Tilt-Zoom (PTZ) Control**: Integrated ONVIF PTZ motor controller with D-pad pan/tilt, continuous movement, and preset zoom.
- **Multiple Viewing Interfaces**:
  - **Desktop GUI**: Native Tkinter + OpenCV interactive desktop viewer.
  - **Web Dashboard**: Modern HTML5/JS web console with dynamic AI toggles, confidence slider, and touch-friendly PTZ controls.
  - **Multi-Stream Hub**: Grid-based multi-camera monitoring console managing concurrent RTSP streams.
- **Low-Latency Streaming**: Optimized snapshot and RTSP pipeline with configurable network caching.

---

## 📁 Repository Structure

```text
camate-ai-vision-controller/
├── camate_viewer.py          # Native desktop Tkinter GUI viewer with PTZ & YOLOv8
├── camate_web_viewer.py      # Embedded HTTP server & interactive web control console
├── camate_multi_stream.py    # Multi-camera grid dashboard for concurrent feeds
├── watch_camate.sh           # Low-latency VLC RTSP feed launcher
├── requirements.txt          # Python package requirements
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/entusedglue3125/camate-ai-vision-controller.git
cd camate-ai-vision-controller

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the Applications

#### Option A: Web Dashboard (Browser Control)
```bash
python3 camate_web_viewer.py
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser to view the live video feed and control camera movement.

#### Option B: Desktop GUI
```bash
python3 camate_viewer.py
```

#### Option C: Multi-Camera Grid
```bash
python3 camate_multi_stream.py
```

---

## ⚙️ Configuration

Update `CAMERA_IP` and credentials inside the scripts to match your local IP camera configuration:
```python
CAMERA_IP = "10.46.187.6"
HTTP_URL = f"http://{CAMERA_IP}/snapshot.jpg"
ONVIF_PTZ_URL = f"http://{CAMERA_IP}:8888/onvif/ptz_service"
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
