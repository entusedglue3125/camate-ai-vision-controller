#!/home/chakradhar/robotics/isaac_environment/.venv/bin/python
"""
Camate Camera Desktop Viewer & Controller with AI Detection
Camera IP: 10.46.187.6
Real-time YOLOv8 Object Detection (GPU) + Face Detection + PTZ motor controls.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import urllib.request
import threading
import time
import io
import os
import cv2
import numpy as np
import base64
import datetime
from ultralytics import YOLO

CAMERA_IP = "10.46.187.6"
HTTP_URL = f"http://{CAMERA_IP}/snapshot.jpg"
ONVIF_PTZ_URL = f"http://{CAMERA_IP}:8888/onvif/ptz_service"
AUTH_HEADER = "Basic " + base64.b64encode(b"admin:").decode()

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

def send_ptz_soap(cmd_xml):
    soap = f"""<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <soap:Header>{make_wsse()}</soap:Header>
  <soap:Body>{cmd_xml}</soap:Body>
</soap:Envelope>"""
    try:
        req = urllib.request.Request(ONVIF_PTZ_URL, data=soap.encode(), headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return True, resp.read().decode()
    except Exception as e:
        return False, str(e)

def ptz_continuous_move(pan, tilt, zoom):
    cmd = f"""<tptz:ContinuousMove>
  <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
  <tptz:Velocity>
    <tt:PanTilt x="{pan}" y="{tilt}"/>
    <tt:Zoom x="{zoom}"/>
  </tptz:Velocity>
</tptz:ContinuousMove>"""
    return send_ptz_soap(cmd)

def ptz_stop():
    cmd = """<tptz:Stop>
  <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
  <tptz:PanTilt>true</tptz:PanTilt>
  <tptz:Zoom>true</tptz:Zoom>
</tptz:Stop>"""
    return send_ptz_soap(cmd)

class CamateViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camate Camera - Real-Time AI & PTZ")
        self.root.geometry("1100x860")
        self.root.configure(bg="#090d16")

        self.running = True
        self.current_raw_frame = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.active_move = False

        # Load AI models
        self.yolo = YOLO("yolov8n.pt")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.detect_objects_var = tk.BooleanVar(value=True)
        self.detect_faces_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._bind_keys()

        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # Header bar
        header = tk.Frame(self.root, bg="#131b2e", pady=10, padx=20)
        header.pack(fill=tk.X)

        title = tk.Label(header, text="Camate AI Vision & PTZ", font=("Helvetica", 16, "bold"), fg="#38bdf8", bg="#131b2e")
        title.pack(side=tk.LEFT)

        self.lbl_status = tk.Label(header, text="● LIVE (RTX 4050)", font=("Helvetica", 11, "bold"), fg="#34d399", bg="#131b2e")
        self.lbl_status.pack(side=tk.RIGHT)

        # Video container
        self.video_container = tk.Frame(self.root, bg="#000000", bd=1, relief=tk.SOLID)
        self.video_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.lbl_video = tk.Label(self.video_container, bg="#000000", text="Starting AI stream...", fg="#94a3b8")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # Bottom Bar: Detection Summary
        self.lbl_det_summary = tk.Label(self.root, text="🎯 Detections: Initializing...", font=("Helvetica", 10),
                                        fg="#cbd5e1", bg="#1e293b", anchor="w", padx=20, pady=5)
        self.lbl_det_summary.pack(fill=tk.X, padx=20)

        # Controls panel at bottom
        panel = tk.Frame(self.root, bg="#131b2e", pady=12, padx=20)
        panel.pack(fill=tk.X, padx=20, pady=(6, 16))

        # Left: AI Options & Snapshot
        left_box = tk.Frame(panel, bg="#131b2e")
        left_box.pack(side=tk.LEFT, fill=tk.Y)

        btn_snap = tk.Button(left_box, text="📸 Capture Snapshot", command=self.save_snapshot,
                             bg="#0284c7", fg="white", font=("Helvetica", 10, "bold"),
                             padx=12, pady=6, relief=tk.FLAT, cursor="hand2")
        btn_snap.pack(anchor="w", pady=(0, 6))

        ai_toggles = tk.Frame(left_box, bg="#131b2e")
        ai_toggles.pack(anchor="w")

        tk.Checkbutton(ai_toggles, text="Object Detection (YOLO)", variable=self.detect_objects_var,
                       fg="#f8fafc", bg="#131b2e", selectcolor="#0ea5e9", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(ai_toggles, text="Face Detection", variable=self.detect_faces_var,
                       fg="#f8fafc", bg="#131b2e", selectcolor="#f43f5e", font=("Helvetica", 9)).pack(side=tk.LEFT)

        tk.Label(left_box, text="Keyboard: Arrow Keys (Move) | +/- (Zoom) | Space (Stop)",
                 font=("Helvetica", 9), fg="#64748b", bg="#131b2e").pack(anchor="w", pady=(6, 0))

        # Right: PTZ D-Pad
        dpad_frame = tk.Frame(panel, bg="#131b2e")
        dpad_frame.pack(side=tk.RIGHT, padx=10)

        btn_opts = {"bg": "#1e293b", "fg": "#f1f5f9", "font": ("Helvetica", 10, "bold"), "width": 4, "height": 1, "relief": tk.FLAT, "cursor": "hand2"}

        b_up = tk.Button(dpad_frame, text="▲", **btn_opts)
        b_up.grid(row=0, column=1, padx=2, pady=2)
        self._bind_hold(b_up, 0.0, 0.4, 0.0)

        b_left = tk.Button(dpad_frame, text="◄", **btn_opts)
        b_left.grid(row=1, column=0, padx=2, pady=2)
        self._bind_hold(b_left, -0.4, 0.0, 0.0)

        b_stop = tk.Button(dpad_frame, text="■", bg="#ef4444", fg="white", font=("Helvetica", 9, "bold"), width=4, height=1, relief=tk.FLAT, cursor="hand2", command=self.do_stop)
        b_stop.grid(row=1, column=1, padx=2, pady=2)

        b_right = tk.Button(dpad_frame, text="►", **btn_opts)
        b_right.grid(row=1, column=2, padx=2, pady=2)
        self._bind_hold(b_right, 0.4, 0.0, 0.0)

        b_down = tk.Button(dpad_frame, text="▼", **btn_opts)
        b_down.grid(row=2, column=1, padx=2, pady=2)
        self._bind_hold(b_down, 0.0, -0.4, 0.0)

        b_zin = tk.Button(dpad_frame, text="🔍+", **btn_opts)
        b_zin.grid(row=0, column=3, padx=(10, 2), pady=2)
        self._bind_hold(b_zin, 0.0, 0.0, 0.4)

        b_zout = tk.Button(dpad_frame, text="🔍-", **btn_opts)
        b_zout.grid(row=2, column=3, padx=(10, 2), pady=2)
        self._bind_hold(b_zout, 0.0, 0.0, -0.4)

    def _bind_hold(self, widget, pan, tilt, zoom):
        def on_press(e):
            self.active_move = True
            threading.Thread(target=ptz_continuous_move, args=(pan, tilt, zoom), daemon=True).start()
        def on_release(e):
            if self.active_move:
                self.active_move = False
                threading.Thread(target=ptz_stop, daemon=True).start()
        widget.bind("<ButtonPress-1>", on_press)
        widget.bind("<ButtonRelease-1>", on_release)

    def _bind_keys(self):
        self.root.bind("<KeyPress-Up>", lambda e: self._key_move(0.0, 0.4, 0.0))
        self.root.bind("<KeyPress-Down>", lambda e: self._key_move(0.0, -0.4, 0.0))
        self.root.bind("<KeyPress-Left>", lambda e: self._key_move(-0.4, 0.0, 0.0))
        self.root.bind("<KeyPress-Right>", lambda e: self._key_move(0.4, 0.0, 0.0))
        self.root.bind("<KeyPress-plus>", lambda e: self._key_move(0.0, 0.0, 0.4))
        self.root.bind("<KeyPress-equal>", lambda e: self._key_move(0.0, 0.0, 0.4))
        self.root.bind("<KeyPress-minus>", lambda e: self._key_move(0.0, 0.0, -0.4))
        self.root.bind("<KeyRelease-Up>", lambda e: self._key_stop())
        self.root.bind("<KeyRelease-Down>", lambda e: self._key_stop())
        self.root.bind("<KeyRelease-Left>", lambda e: self._key_stop())
        self.root.bind("<KeyRelease-Right>", lambda e: self._key_stop())
        self.root.bind("<KeyRelease-plus>", lambda e: self._key_stop())
        self.root.bind("<KeyRelease-equal>", lambda e: self._key_stop())
        self.root.bind("<KeyRelease-minus>", lambda e: self._key_stop())
        self.root.bind("<space>", lambda e: self.do_stop())

    def _key_move(self, pan, tilt, zoom):
        if not self.active_move:
            self.active_move = True
            threading.Thread(target=ptz_continuous_move, args=(pan, tilt, zoom), daemon=True).start()

    def _key_stop(self):
        if self.active_move:
            self.active_move = False
            threading.Thread(target=ptz_stop, daemon=True).start()

    def do_stop(self):
        self.active_move = False
        threading.Thread(target=ptz_stop, daemon=True).start()

    def save_snapshot(self):
        if not self.current_raw_frame:
            return
        os.makedirs(os.path.expanduser("~/Pictures"), exist_ok=True)
        filename = os.path.expanduser(f"~/Pictures/Camate_AI_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        with open(filename, "wb") as f:
            f.write(self.current_raw_frame)
        messagebox.showinfo("Saved", f"Snapshot saved to:\n{filename}")

    def _stream_loop(self):
        while self.running:
            try:
                req = urllib.request.Request(HTTP_URL, headers={"Authorization": AUTH_HEADER})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    raw_data = resp.read()

                arr = np.frombuffer(raw_data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    time.sleep(0.05)
                    continue

                detections = []

                # 1. YOLO Object Detection
                if self.detect_objects_var.get():
                    results = self.yolo(img, device="cuda:0", conf=0.35, verbose=False)[0]
                    for box in results.boxes:
                        cls_name = results.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        detections.append(cls_name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = (0, 230, 118) if cls_name == "person" else (0, 180, 216)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {int(conf*100)}%"
                        cv2.putText(img, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                # 2. Face Detection
                if self.detect_faces_var.get():
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
                    for (fx, fy, fw, fh) in faces:
                        detections.append("face")
                        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (244, 63, 94), 2)
                        cv2.putText(img, "Face", (fx + 2, max(14, fy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (244, 63, 94), 1, cv2.LINE_AA)

                # Summary counts
                counts = {}
                for d in detections:
                    counts[d] = counts.get(d, 0) + 1
                det_text = "🎯 Detections: " + (", ".join(f"{v} {k}" for k, v in counts.items()) if counts else "None in view")

                # Convert BGR to RGB for Tkinter
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                vw = self.video_container.winfo_width()
                vh = self.video_container.winfo_height()
                if vw > 100 and vh > 100:
                    pil_img.thumbnail((vw, vh), Image.Resampling.BILINEAR)

                photo = ImageTk.PhotoImage(pil_img)

                # Encode current annotated frame for snapshot button
                _, enc = cv2.imencode(".jpg", img)
                self.current_raw_frame = enc.tobytes()

                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now

                def update(p=photo, f=self.fps, dt=det_text):
                    if not self.running: return
                    self.lbl_video.configure(image=p, text="")
                    self.lbl_video.image = p
                    self.lbl_status.configure(text=f"● LIVE | {f:.1f} FPS (GPU)")
                    self.lbl_det_summary.configure(text=dt)

                self.root.after(0, update)
                time.sleep(0.04)
            except Exception as e:
                time.sleep(0.3)

    def on_close(self):
        self.running = False
        ptz_stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CamateViewerApp(root)
    root.mainloop()
