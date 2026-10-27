#!/home/chakradhar/robotics/isaac_environment/.venv/bin/python
"""
Multi-Stream Camera Hub with Real-Time AI & PTZ Control
Supports multiple IP cameras in a unified multi-view grid.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import base64
import time
import os
import cv2
import numpy as np
import threading
import datetime
from ultralytics import YOLO

# Global AI Models
print("Loading YOLOv8 and Face Cascade on GPU...")
yolo_model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("AI Models loaded successfully!")

# Global State
ai_settings = {
    "detect_objects": True,
    "detect_faces": True,
    "confidence": 0.35,
}

# Camera Registry: Dict of id -> dict
cameras = {
    "cam1": {
        "id": "cam1",
        "name": "Camate Camera 1",
        "ip": "10.46.187.6",
        "auth": "Basic " + base64.b64encode(b"admin:").decode(),
        "ptz_port": 8888,
        "latest_frame": None,
        "detections": [],
        "fps": 0,
        "status": "Connecting",
        "lock": threading.Lock()
    }
}

active_ptz_cam_id = "cam1"
reg_lock = threading.Lock()

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

def ptz_move(cam_id, pan, tilt, zoom):
    with reg_lock:
        cam = cameras.get(cam_id)
    if not cam:
        return False, "Camera not found"
    ip = cam["ip"]
    port = cam.get("ptz_port", 8888)
    url = f"http://{ip}:{port}/onvif/ptz_service"

    body = f"""<tptz:ContinuousMove xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
  <tptz:Velocity>
    <tt:PanTilt x="{pan}" y="{tilt}"/>
    <tt:Zoom x="{zoom}"/>
  </tptz:Velocity>
</tptz:ContinuousMove>"""
    soap = f"""<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Header>{make_wsse()}</soap:Header>
  <soap:Body>{body}</soap:Body>
</soap:Envelope>"""
    try:
        req = urllib.request.Request(url, data=soap.encode(), headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return True, "OK"
    except Exception as e:
        return False, str(e)

def ptz_stop(cam_id):
    with reg_lock:
        cam = cameras.get(cam_id)
    if not cam:
        return False, "Camera not found"
    ip = cam["ip"]
    port = cam.get("ptz_port", 8888)
    url = f"http://{ip}:{port}/onvif/ptz_service"

    body = """<tptz:Stop xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
  <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
  <tptz:PanTilt>true</tptz:PanTilt>
  <tptz:Zoom>true</tptz:Zoom>
</tptz:Stop>"""
    soap = f"""<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Header>{make_wsse()}</soap:Header>
  <soap:Body>{body}</soap:Body>
</soap:Envelope>"""
    try:
        req = urllib.request.Request(url, data=soap.encode(), headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return True, "OK"
    except Exception as e:
        return False, str(e)

def camera_worker(cam_id):
    print(f"Starting worker for camera {cam_id}...")
    frame_count = 0
    last_fps_time = time.time()

    while True:
        with reg_lock:
            cam = cameras.get(cam_id)
        if not cam:
            break

        ip = cam["ip"]
        auth = cam["auth"]
        url = f"http://{ip}/snapshot.jpg"

        try:
            req = urllib.request.Request(url, headers={"Authorization": auth})
            with urllib.request.urlopen(req, timeout=2) as resp:
                raw_bytes = resp.read()

            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is not None:
                det_list = []

                # 1. Object Detection (YOLOv8 on GPU)
                if ai_settings["detect_objects"]:
                    conf = ai_settings["confidence"]
                    results = yolo_model(img, device="cuda:0", conf=conf, verbose=False)[0]
                    for box in results.boxes:
                        cls_name = results.names[int(box.cls[0])]
                        c = float(box.conf[0])
                        det_list.append(cls_name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = (0, 230, 118) if cls_name == "person" else (0, 180, 216)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {int(c*100)}%"
                        cv2.putText(img, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                # 2. Face Detection
                if ai_settings["detect_faces"]:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
                    for (fx, fy, fw, fh) in faces:
                        det_list.append("face")
                        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (244, 63, 94), 2)
                        cv2.putText(img, "Face", (fx + 2, max(14, fy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (244, 63, 94), 1, cv2.LINE_AA)

                # FPS calculation
                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    cam["fps"] = round(frame_count / (now - last_fps_time), 1)
                    frame_count = 0
                    last_fps_time = now

                cam["status"] = f"LIVE ({cam['fps']} FPS)"
                cam["detections"] = det_list

                # Encode JPEG
                _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with cam["lock"]:
                    cam["latest_frame"] = enc.tobytes()

                time.sleep(0.03)
            else:
                cam["status"] = "No Image"
                time.sleep(0.5)

        except Exception as e:
            cam["status"] = f"Offline ({e})"
            time.sleep(1.0)

# Start initial worker
threading.Thread(target=camera_worker, args=("cam1",), daemon=True).start()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Multi-Stream AI Camera Hub</title>
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
      max-width: 1400px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
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
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .btn-add {
      background: #059669;
      color: white;
      border: none;
      padding: 7px 14px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      font-size: 13px;
    }
    .btn-add:hover { background: #047857; }
    
    .layout-container {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 18px;
      width: 100%;
      max-width: 1400px;
    }
    @media (max-width: 1024px) {
      .layout-container { grid-template-columns: 1fr; }
    }

    /* Streams Grid */
    .streams-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }
    .stream-tile {
      background: #000;
      border: 2px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
      transition: border-color 0.2s;
    }
    .stream-tile.active-ptz {
      border-color: var(--accent);
      box-shadow: 0 0 15px rgba(14, 165, 233, 0.3);
    }
    .stream-header {
      padding: 8px 12px;
      background: var(--card-bg);
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
    }
    .cam-name {
      font-weight: 700;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .cam-badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 9999px;
      background: #064e3b;
      color: #34d399;
      font-weight: 600;
    }
    .stream-view {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .stream-view img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .stream-footer {
      padding: 8px 12px;
      background: #111827;
      border-top: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
    }
    .det-tag {
      color: #94a3b8;
    }
    .btn-select-ptz {
      background: #1e293b;
      color: #cbd5e1;
      border: 1px solid #334155;
      padding: 4px 10px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 11px;
    }
    .btn-select-ptz:hover { background: #334155; }
    .btn-select-ptz.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }

    /* Right Control Sidebar */
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .panel-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .panel-title {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin: 0;
      font-weight: 700;
    }
    .switch-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
    }
    .switch {
      position: relative;
      display: inline-block;
      width: 40px;
      height: 22px;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155;
      border-radius: 22px;
      transition: .2s;
    }
    .slider:before {
      position: absolute; content: "";
      height: 16px; width: 16px;
      left: 3px; bottom: 3px;
      background-color: white;
      border-radius: 50%;
      transition: .2s;
    }
    input:checked + .slider { background-color: var(--accent); }
    input:checked + .slider:before { transform: translateX(18px); }

    /* D-Pad */
    .dpad-container {
      display: grid;
      grid-template-columns: repeat(3, 56px);
      grid-template-rows: repeat(3, 56px);
      gap: 6px;
      justify-content: center;
      margin: 4px 0;
    }
    .dpad-btn {
      width: 56px;
      height: 56px;
      border-radius: 8px;
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
    }
    .dpad-btn:hover { background: #334155; }
    .dpad-btn:active { background: var(--accent); }
    .dpad-center {
      background: #0f172a;
      font-size: 11px;
      font-weight: 700;
      color: #ef4444;
      border: 1px dashed #475569;
    }
    .dpad-center:hover { background: #ef4444; color: white; }

    .zoom-row { display: flex; gap: 8px; }
    .zoom-btn {
      flex: 1;
      padding: 8px;
      border-radius: 6px;
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }
    .zoom-btn:hover { background: #334155; }
    .zoom-btn:active { background: var(--accent); }

    /* Modal for Adding Camera */
    .modal-overlay {
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.7);
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }
    .modal-overlay.open { display: flex; }
    .modal-box {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      width: 380px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .modal-box h3 { margin: 0; color: #38bdf8; font-size: 16px; }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 13px;
      color: var(--text-muted);
    }
    .form-group input {
      background: #0b1120;
      border: 1px solid #334155;
      color: var(--text);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 14px;
    }
    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 10px;
    }
    .btn-cancel {
      background: #334155;
      color: var(--text);
      border: none;
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
    }
    .btn-save {
      background: var(--accent);
      color: white;
      border: none;
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <header>
    <h1>Multi-Stream AI Camera Hub</h1>
    <div class="header-actions">
      <button class="btn-add" onclick="openAddModal()">➕ Add Camera IP</button>
    </div>
  </header>

  <div class="layout-container">
    <div class="streams-grid" id="streams-container">
      <!-- Camera tiles injected here -->
    </div>

    <div class="sidebar">
      <div class="panel-card">
        <h2 class="panel-title">Active PTZ Target</h2>
        <div style="font-size:13px; color:#cbd5e1;" id="active-ptz-label">Camera: Camate Camera 1</div>
        <div class="dpad-container">
          <div></div>
          <button class="dpad-btn" data-pan="0" data-tilt="1">▲</button>
          <div></div>
          <button class="dpad-btn" data-pan="-1" data-tilt="0">◄</button>
          <button class="dpad-btn dpad-center" id="btn-stop">STOP</button>
          <button class="dpad-btn" data-pan="1" data-tilt="0">►</button>
          <div></div>
          <button class="dpad-btn" data-pan="0" data-tilt="-1">▼</button>
          <div></div>
        </div>
        <div class="zoom-row">
          <button class="zoom-btn" data-zoom="1">🔍 Zoom In</button>
          <button class="zoom-btn" data-zoom="-1">🔍 Zoom Out</button>
        </div>
      </div>

      <div class="panel-card">
        <h2 class="panel-title">AI Vision Settings</h2>
        <div class="switch-row">
          <span>Object Detection (YOLOv8)</span>
          <label class="switch">
            <input type="checkbox" id="chk-obj" checked onchange="updateAi()">
            <span class="slider"></span>
          </label>
        </div>
        <div class="switch-row">
          <span>Face Detection</span>
          <label class="switch">
            <input type="checkbox" id="chk-face" checked onchange="updateAi()">
            <span class="slider"></span>
          </label>
        </div>
      </div>
    </div>
  </div>

  <!-- Add Camera Modal -->
  <div class="modal-overlay" id="add-modal">
    <div class="modal-box">
      <h3>Add New Camera Stream</h3>
      <div class="form-group">
        <label>Camera Name:</label>
        <input type="text" id="new-name" placeholder="e.g. Camate Camera 2">
      </div>
      <div class="form-group">
        <label>IP Address:</label>
        <input type="text" id="new-ip" placeholder="e.g. 10.46.187.7">
      </div>
      <div class="form-group">
        <label>Username (optional):</label>
        <input type="text" id="new-user" value="admin">
      </div>
      <div class="form-group">
        <label>Password (optional):</label>
        <input type="password" id="new-pwd" placeholder="empty if none">
      </div>
      <div class="modal-actions">
        <button class="btn-cancel" onclick="closeAddModal()">Cancel</button>
        <button class="btn-save" onclick="saveNewCamera()">Add Stream</button>
      </div>
    </div>
  </div>

  <script>
    let activePtzId = "cam1";

    function openAddModal() {
      document.getElementById('add-modal').classList.add('open');
    }
    function closeAddModal() {
      document.getElementById('add-modal').classList.remove('open');
    }

    async function saveNewCamera() {
      const name = document.getElementById('new-name').value.trim() || "Camera 2";
      const ip = document.getElementById('new-ip').value.trim();
      const user = document.getElementById('new-user').value.trim();
      const pwd = document.getElementById('new-pwd').value;
      if (!ip) { alert("Please enter camera IP"); return; }

      await fetch(`/api/camera/add?name=${encodeURIComponent(name)}&ip=${encodeURIComponent(ip)}&user=${encodeURIComponent(user)}&pwd=${encodeURIComponent(pwd)}`);
      closeAddModal();
      loadCameras();
    }

    async function deleteCamera(id) {
      if (!confirm("Remove this camera stream?")) return;
      await fetch(`/api/camera/remove?id=${id}`);
      loadCameras();
    }

    async function setPtzTarget(id, name) {
      activePtzId = id;
      document.getElementById('active-ptz-label').innerText = `Camera: ${name}`;
      document.querySelectorAll('.stream-tile').forEach(t => t.classList.remove('active-ptz'));
      const activeTile = document.getElementById(`tile-${id}`);
      if (activeTile) activeTile.classList.add('active-ptz');
      await fetch(`/api/ptz/target?id=${id}`);
    }

    async function updateAi() {
      const obj = document.getElementById('chk-obj').checked;
      const face = document.getElementById('chk-face').checked;
      await fetch(`/api/ai?objects=${obj}&faces=${face}`);
    }

    async function loadCameras() {
      const res = await fetch('/api/cameras');
      const data = await res.json();
      const container = document.getElementById('streams-container');
      container.innerHTML = '';

      data.cameras.forEach(cam => {
        const isPtz = (cam.id === activePtzId);
        const tile = document.createElement('div');
        tile.className = `stream-tile ${isPtz ? 'active-ptz' : ''}`;
        tile.id = `tile-${cam.id}`;
        tile.innerHTML = `
          <div class="stream-header">
            <span class="cam-name">📹 ${cam.name} (${cam.ip})</span>
            <span class="cam-badge" id="badge-${cam.id}">${cam.status}</span>
          </div>
          <div class="stream-view">
            <img src="/stream/${cam.id}" alt="${cam.name}">
          </div>
          <div class="stream-footer">
            <span class="det-tag" id="det-${cam.id}">Detections: Running...</span>
            <div style="display:flex; gap:6px;">
              <button class="btn-select-ptz ${isPtz ? 'active' : ''}" onclick="setPtzTarget('${cam.id}', '${cam.name}')">🎯 PTZ Control</button>
              ${cam.id !== 'cam1' ? `<button class="btn-select-ptz" style="color:#ef4444;" onclick="deleteCamera('${cam.id}')">✕</button>` : ''}
            </div>
          </div>
        `;
        container.appendChild(tile);
      });
    }

    // Live detection polling
    setInterval(async () => {
      try {
        const res = await fetch('/api/cameras');
        const data = await res.json();
        data.cameras.forEach(cam => {
          const badge = document.getElementById(`badge-${cam.id}`);
          if (badge) badge.innerText = cam.status;
          const det = document.getElementById(`det-${cam.id}`);
          if (det) {
            const counts = {};
            (cam.detections || []).forEach(x => { counts[x] = (counts[x] || 0) + 1; });
            const s = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ');
            det.innerText = s ? `🎯 ${s}` : '🎯 None in view';
          }
        });
      } catch (e) {}
    }, 1200);

    // PTZ Controls
    let isMoving = false;
    async function sendPtz(action, pan=0, tilt=0, zoom=0) {
      try {
        await fetch(`/api/ptz?cam_id=${activePtzId}&action=${action}&pan=${pan}&tilt=${tilt}&zoom=${zoom}`);
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

    window.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowUp') { e.preventDefault(); startMove(0, 1); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); startMove(0, -1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); startMove(-1, 0); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); startMove(1, 0); }
      else if (e.key === '+' || e.key === '=') { e.preventDefault(); startMove(0, 0, 1); }
      else if (e.key === '-' || e.key === '_') { e.preventDefault(); startMove(0, 0, -1); }
      else if (e.key === ' ') { e.preventDefault(); stopMove(); }
    });
    window.addEventListener('keyup', (e) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', '+', '=', '-', '_'].includes(e.key)) stopMove();
    });

    loadCameras();
  </script>
</body>
</html>
"""

class MultiCamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path
        query = urllib.parse.parse_qs(url_parts.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path.startswith("/stream/"):
            cam_id = path.split("/stream/")[1]
            with reg_lock:
                cam = cameras.get(cam_id)
            if not cam:
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                last_frame = None
                while True:
                    with cam["lock"]:
                        f = cam["latest_frame"]
                    if f and f != last_frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(f)}\r\n\r\n".encode())
                        self.wfile.write(f)
                        self.wfile.write(b"\r\n")
                        last_frame = f
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path == "/api/cameras":
            cam_list = []
            with reg_lock:
                for cid, c in cameras.items():
                    cam_list.append({
                        "id": c["id"],
                        "name": c["name"],
                        "ip": c["ip"],
                        "status": c["status"],
                        "detections": c.get("detections", []),
                        "fps": c.get("fps", 0)
                    })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"cameras": cam_list}).encode("utf-8"))

        elif path == "/api/camera/add":
            name = query.get("name", ["Camera 2"])[0]
            ip = query.get("ip", [""])[0]
            user = query.get("user", ["admin"])[0]
            pwd = query.get("pwd", [""])[0]

            if ip:
                auth = "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()
                new_id = f"cam_{int(time.time())}"
                new_cam = {
                    "id": new_id,
                    "name": name,
                    "ip": ip,
                    "auth": auth,
                    "ptz_port": 8888,
                    "latest_frame": None,
                    "detections": [],
                    "fps": 0,
                    "status": "Connecting",
                    "lock": threading.Lock()
                }
                with reg_lock:
                    cameras[new_id] = new_cam
                threading.Thread(target=camera_worker, args=(new_id,), daemon=True).start()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "id": new_id}).encode("utf-8"))
            else:
                self.send_response(400)
                self.end_headers()

        elif path == "/api/camera/remove":
            cid = query.get("id", [""])[0]
            with reg_lock:
                if cid in cameras and cid != "cam1":
                    del cameras[cid]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))

        elif path == "/api/ptz/target":
            global active_ptz_cam_id
            cid = query.get("id", ["cam1"])[0]
            active_ptz_cam_id = cid
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"active": active_ptz_cam_id}).encode("utf-8"))

        elif path == "/api/ptz":
            cid = query.get("cam_id", [active_ptz_cam_id])[0]
            action = query.get("action", ["stop"])[0]
            pan = float(query.get("pan", [0.0])[0])
            tilt = float(query.get("tilt", [0.0])[0])
            zoom = float(query.get("zoom", [0.0])[0])

            if action == "continuous":
                ok, msg = ptz_move(cid, pan, tilt, zoom)
            else:
                ok, msg = ptz_stop(cid)

            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "message": msg}).encode("utf-8"))

        elif path == "/api/ai":
            if "objects" in query:
                ai_settings["detect_objects"] = (query["objects"][0].lower() == "true")
            if "faces" in query:
                ai_settings["detect_faces"] = (query["faces"][0].lower() == "true")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "settings": ai_settings}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8088), MultiCamHandler)
    print("Multi-Stream AI Camera Hub running at http://localhost:8088")
    server.serve_forever()
