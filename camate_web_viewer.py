#!/home/chakradhar/robotics/isaac_environment/.venv/bin/python
"""
Camate Camera Live AI Vision & PTZ Controller
Camera IP: 10.46.187.6
Features:
 - Real-time Object Detection (YOLOv8 on NVIDIA RTX 4050 GPU)
 - Real-time Face Detection (OpenCV Face Cascade)
 - Live Web Dashboard with D-Pad & PTZ Motor Controls
 - Dynamic AI Toggles (Face on/off, Objects on/off, Confidence slider)
 - Low-latency MJPEG Stream
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import base64
import time
import os
import io
import cv2
import numpy as np
import threading
import datetime
from ultralytics import YOLO

CAMERA_IP = "10.46.187.6"
HTTP_SNAPSHOT_URL = f"http://{CAMERA_IP}/snapshot.jpg"
ONVIF_PTZ_URL = f"http://{CAMERA_IP}:8888/onvif/ptz_service"
AUTH_HEADER = "Basic " + base64.b64encode(b"admin:").decode()

# Global AI Settings
ai_settings = {
    "detect_objects": True,
    "detect_faces": True,
    "confidence": 0.35,
    "latest_detections": []
}

# Load Models
print("Loading AI Models (YOLOv8 + Face Cascade)...")
yolo_model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("AI Models loaded successfully on GPU!")

def make_wsse(username="admin", password=""):
    created = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    nonce_raw = os.urandom(16)
    nonce_b64 = base64.b64encode(nonce_raw).decode()
    combined = nonce_raw + created.encode("utf-8") + password.encode("utf-8")
    digest = base64.b64encode(__import__("hashlib").sha1(combined).digest()).decode()
    return f"""<wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">
  <wsse:UsernameToken>
    <wsse:Username>{username}</wsse:Username>
    <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</wsse:Password>
    <wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce_b64}</wsse:Nonce>
    <wsu:Created>{created}</wsu:Created>
  </wsse:UsernameToken>
</wsse:Security>"""

def send_onvif_ptz(body_xml):
    soap = f"""<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <soap:Header>{make_wsse()}</soap:Header>
  <soap:Body>{body_xml}</soap:Body>
</soap:Envelope>"""
    try:
        req = urllib.request.Request(ONVIF_PTZ_URL, data=soap.encode(), headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return True, resp.read().decode()
    except Exception as e:
        return False, str(e)

def ptz_continuous_move(pan, tilt, zoom):
    body = f"""<tptz:ContinuousMove>
  <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
  <tptz:Velocity>
    <tt:PanTilt x="{pan}" y="{tilt}"/>
    <tt:Zoom x="{zoom}"/>
  </tptz:Velocity>
</tptz:ContinuousMove>"""
    return send_onvif_ptz(body)

def ptz_stop():
    body = """<tptz:Stop>
  <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
  <tptz:PanTilt>true</tptz:PanTilt>
  <tptz:Zoom>true</tptz:Zoom>
</tptz:Stop>"""
    return send_onvif_ptz(body)

# Frame Grabber & AI Processor Thread
latest_processed_frame = None
frame_lock = threading.Lock()

def video_processor_loop():
    global latest_processed_frame
    while True:
        try:
            req = urllib.request.Request(HTTP_SNAPSHOT_URL, headers={"Authorization": AUTH_HEADER})
            with urllib.request.urlopen(req, timeout=2) as resp:
                raw_bytes = resp.read()

            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                time.sleep(0.05)
                continue

            h, w = img.shape[:2]
            detections_summary = []

            # 1. Object Detection via YOLOv8 (GPU)
            if ai_settings["detect_objects"]:
                conf_thresh = ai_settings["confidence"]
                results = yolo_model(img, device="cuda:0", conf=conf_thresh, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = results.names[cls_id]
                    detections_summary.append(cls_name)

                    # Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Color scheme per class type
                    if cls_name == "person":
                        color = (0, 230, 118)  # Vibrant Emerald Green
                    elif cls_name in ["car", "bus", "truck", "motorcycle"]:
                        color = (255, 171, 0)  # Amber
                    else:
                        color = (0, 180, 216)  # Cyan Blue

                    # Draw clean rounded-like rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {int(conf*100)}%"

                    # Label background
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), color, -1)
                    cv2.putText(img, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # 2. Face Detection via OpenCV Cascade
            if ai_settings["detect_faces"]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(32, 32))
                for (fx, fy, fw, fh) in faces:
                    detections_summary.append("face")
                    face_color = (244, 63, 94) # Rose / Pink
                    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), face_color, 2)

                    flabel = "Face"
                    (tw, th), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (fx, max(0, fy - 20)), (fx + tw + 6, max(0, fy)), face_color, -1)
                    cv2.putText(img, flabel, (fx + 3, max(14, fy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Update detection summary
            ai_settings["latest_detections"] = detections_summary

            # Encode to JPEG
            _, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with frame_lock:
                latest_processed_frame = jpeg_buf.tobytes()

            time.sleep(0.03) # Smooth ~20 FPS loop
        except Exception as e:
            time.sleep(0.2)

# Start background video processor
proc_thread = threading.Thread(target=video_processor_loop, daemon=True)
proc_thread.start()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Camate Camera - Real-Time AI Detection & PTZ</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #131b2e;
      --card-border: #1e293b;
      --accent: #0ea5e9;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 16px;
    }
    header {
      width: 100%;
      max-width: 1140px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #064e3b;
      color: #34d399;
      font-size: 13px;
      padding: 4px 12px;
      border-radius: 9999px;
      font-weight: 600;
    }
    .badge-dot {
      width: 8px;
      height: 8px;
      background: #34d399;
      border-radius: 50%;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%{opacity:1;} 50%{opacity:0.3;} 100%{opacity:1;} }

    .main-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 18px;
      width: 100%;
      max-width: 1140px;
    }
    @media (max-width: 880px) {
      .main-grid { grid-template-columns: 1fr; }
    }
    .video-card {
      background: #000;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
    .video-container {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .detection-banner {
      background: #1e293b;
      padding: 8px 14px;
      font-size: 13px;
      border-top: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      gap: 8px;
      color: #cbd5e1;
    }
    .panel-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .panel-title {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin: 0;
      font-weight: 700;
    }
    .toggle-group {
      background: #0b1120;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .toggle-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
    }
    .switch {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155;
      transition: .2s;
      border-radius: 24px;
    }
    .slider:before {
      position: absolute; content: "";
      height: 18px; width: 18px;
      left: 3px; bottom: 3px;
      background-color: white;
      transition: .2s;
      border-radius: 50%;
    }
    input:checked + .slider { background-color: var(--accent); }
    input:checked + .slider:before { transform: translateX(20px); }

    .range-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 13px;
      color: var(--text-muted);
    }
    .range-row input {
      width: 100%;
      accent-color: var(--accent);
    }

    .dpad-container {
      display: grid;
      grid-template-columns: repeat(3, 62px);
      grid-template-rows: repeat(3, 62px);
      gap: 6px;
      justify-content: center;
      margin: 4px 0;
    }
    .dpad-btn {
      width: 62px;
      height: 62px;
      border-radius: 8px;
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 18px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      transition: all 0.1s;
    }
    .dpad-btn:hover { background: #334155; }
    .dpad-btn:active { background: var(--accent); transform: scale(0.95); }
    .dpad-center {
      background: #0f172a;
      font-size: 12px;
      font-weight: 700;
      color: #ef4444;
      border: 1px dashed #475569;
    }
    .dpad-center:hover { background: #ef4444; color: white; }

    .zoom-row {
      display: flex;
      gap: 8px;
    }
    .zoom-btn {
      flex: 1;
      padding: 9px;
      border-radius: 8px;
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      user-select: none;
    }
    .zoom-btn:hover { background: #334155; }
    .zoom-btn:active { background: var(--accent); }

    .btn-action {
      background: var(--accent);
      color: white;
      border: none;
      padding: 9px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      text-align: center;
      display: block;
    }
    .btn-action:hover { background: var(--accent-hover); }

    .info-footer {
      grid-column: 1 / -1;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 13px;
      color: var(--text-muted);
    }
    code {
      background: #0b1120;
      color: #38bdf8;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <header>
    <h1>Camate Live AI Vision & PTZ</h1>
    <div class="badge">
      <div class="badge-dot"></div>
      RTX 4050 GPU ACCELERATED
    </div>
  </header>

  <div class="main-grid">
    <div class="video-card">
      <div class="video-container">
        <img id="stream" src="/live.mjpeg" alt="Live AI Camera Feed">
      </div>
      <div class="detection-banner">
        <span>🎯 <strong>Detections:</strong></span>
        <span id="detection-text">Running inference...</span>
      </div>
    </div>

    <div class="panel-card">
      <h2 class="panel-title">AI Vision Settings</h2>
      <div class="toggle-group">
        <div class="toggle-row">
          <span>Object Detection (YOLOv8)</span>
          <label class="switch">
            <input type="checkbox" id="chk-objects" checked onchange="updateAi()">
            <span class="slider"></span>
          </label>
        </div>

        <div class="toggle-row">
          <span>Face Detection</span>
          <label class="switch">
            <input type="checkbox" id="chk-faces" checked onchange="updateAi()">
            <span class="slider"></span>
          </label>
        </div>

        <div class="range-row">
          <div style="display:flex; justify-content:space-between;">
            <span>Confidence Threshold:</span>
            <span id="conf-val">35%</span>
          </div>
          <input type="range" id="rng-conf" min="15" max="85" value="35" oninput="updateConf(this.value)">
        </div>
      </div>

      <h2 class="panel-title">Camera Movement (PTZ)</h2>
      <div class="dpad-container">
        <div></div>
        <button class="dpad-btn" data-pan="0" data-tilt="1" title="Tilt Up">▲</button>
        <div></div>

        <button class="dpad-btn" data-pan="-1" data-tilt="0" title="Pan Left">◄</button>
        <button class="dpad-btn dpad-center" id="btn-stop">STOP</button>
        <button class="dpad-btn" data-pan="1" data-tilt="0" title="Pan Right">►</button>

        <div></div>
        <button class="dpad-btn" data-pan="0" data-tilt="-1" title="Tilt Down">▼</button>
        <div></div>
      </div>

      <div class="zoom-row">
        <button class="zoom-btn" data-zoom="1">🔍 Zoom In (+)</button>
        <button class="zoom-btn" data-zoom="-1">🔍 Zoom Out (-)</button>
      </div>

      <a href="/snapshot" target="_blank" download class="btn-action">📸 Save Current Frame</a>
    </div>

    <div class="info-footer">
      <strong>Stream Source:</strong> <code>rtsp://admin:@10.46.187.6:554/ch0_0.264</code> |
      <strong>Keyboard Controls:</strong> Arrow Keys (Pan/Tilt), +/- (Zoom), Space (Stop)
    </div>
  </div>

  <script>
    let isMoving = false;

    async function updateAi() {
      const obj = document.getElementById('chk-objects').checked;
      const face = document.getElementById('chk-faces').checked;
      const conf = parseFloat(document.getElementById('rng-conf').value) / 100.0;
      await fetch(`/api/ai?objects=${obj}&faces=${face}&conf=${conf}`);
    }

    function updateConf(val) {
      document.getElementById('conf-val').innerText = val + '%';
      updateAi();
    }

    // Poll live detections
    setInterval(async () => {
      try {
        const res = await fetch('/api/detections');
        const data = await res.json();
        const counts = {};
        data.detections.forEach(x => { counts[x] = (counts[x] || 0) + 1; });
        const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}${v > 1 ? 's' : ''}`).join(', ');
        document.getElementById('detection-text').innerText = summary || 'None in view';
      } catch (e) {}
    }, 1000);

    // PTZ Controls
    async function sendPtz(action, pan=0, tilt=0, zoom=0) {
      try {
        await fetch(`/api/ptz?action=${action}&pan=${pan}&tilt=${tilt}&zoom=${zoom}`);
      } catch (e) {}
    }

    function startMove(pan, tilt, zoom=0) {
      isMoving = true;
      sendPtz('continuous', pan * 0.4, tilt * 0.4, zoom * 0.4);
    }

    function stopMove() {
      if (isMoving) {
        isMoving = false;
        sendPtz('stop');
      }
    }

    document.querySelectorAll('.dpad-btn[data-pan], .zoom-btn[data-zoom]').forEach(btn => {
      const pan = parseFloat(btn.dataset.pan || 0);
      const tilt = parseFloat(btn.dataset.tilt || 0);
      const zoom = parseFloat(btn.dataset.zoom || 0);

      btn.addEventListener('mousedown', (e) => { e.preventDefault(); startMove(pan, tilt, zoom); });
      btn.addEventListener('mouseup', (e) => { e.preventDefault(); stopMove(); });
      btn.addEventListener('mouseleave', () => { if (isMoving) stopMove(); });

      btn.addEventListener('touchstart', (e) => { e.preventDefault(); startMove(pan, tilt, zoom); });
      btn.addEventListener('touchend', (e) => { e.preventDefault(); stopMove(); });
    });

    document.getElementById('btn-stop').addEventListener('click', stopMove);

    // Keyboard Shortcuts
    const activeKeys = new Set();
    window.addEventListener('keydown', (e) => {
      if (activeKeys.has(e.key)) return;
      activeKeys.add(e.key);
      if (e.key === 'ArrowUp') { e.preventDefault(); startMove(0, 1); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); startMove(0, -1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); startMove(-1, 0); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); startMove(1, 0); }
      else if (e.key === '+' || e.key === '=') { e.preventDefault(); startMove(0, 0, 1); }
      else if (e.key === '-' || e.key === '_') { e.preventDefault(); startMove(0, 0, -1); }
      else if (e.key === ' ') { e.preventDefault(); stopMove(); }
    });

    window.addEventListener('keyup', (e) => {
      activeKeys.delete(e.key);
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', '+', '=', '-', '_'].includes(e.key)) {
        stopMove();
      }
    });
  </script>
</body>
</html>
"""

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path
        query = urllib.parse.parse_qs(url_parts.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/snapshot":
            with frame_lock:
                frame = latest_processed_frame
            if frame:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(503)
                self.end_headers()

        elif path == "/live.mjpeg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last_sent = None
                while True:
                    with frame_lock:
                        frame = latest_processed_frame
                    if frame and frame != last_sent:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        last_sent = frame
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path == "/api/ai":
            if "objects" in query:
                ai_settings["detect_objects"] = (query["objects"][0].lower() == "true")
            if "faces" in query:
                ai_settings["detect_faces"] = (query["faces"][0].lower() == "true")
            if "conf" in query:
                try:
                    ai_settings["confidence"] = float(query["conf"][0])
                except:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "settings": ai_settings}).encode("utf-8"))

        elif path == "/api/detections":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detections": ai_settings["latest_detections"]}).encode("utf-8"))

        elif path == "/api/ptz":
            action = query.get("action", ["stop"])[0]
            pan = float(query.get("pan", [0.0])[0])
            tilt = float(query.get("tilt", [0.0])[0])
            zoom = float(query.get("zoom", [0.0])[0])

            if action == "continuous":
                ok, _ = ptz_continuous_move(pan, tilt, zoom)
            else:
                ok, _ = ptz_stop()

            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8088), CamHandler)
    print("Camate AI Vision & PTZ Controller running at http://localhost:8088")
    server.serve_forever()
