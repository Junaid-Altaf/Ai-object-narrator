"""
╔══════════════════════════════════════════════════════════════════╗
║          AI-Powered Object Narrator — app.py                    ║
║          Built for visually impaired users                      ║
║          Stack: Streamlit · YOLOv8 · EasyOCR · gTTS            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import io
import threading
import base64
import tempfile
from pathlib import Path
from datetime import datetime

# ── Lazy imports (heavy models loaded only once via st.cache_resource) ──────

@st.cache_resource(show_spinner="Loading YOLO model…")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")          # nano – fastest, still accurate

@st.cache_resource(show_spinner="Loading OCR engine…")
def load_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

def tts_speak(text: str) -> str:
    """Convert text → gTTS → base64 audio data-URI for auto-play in browser."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:audio/mp3;base64,{b64}"
    except Exception as e:
        st.warning(f"TTS error: {e}")
        return ""

def autoplay_audio(data_uri: str):
    """Inject a hidden <audio> element that auto-plays once."""
    if not data_uri:
        return
    audio_html = f"""
    <audio autoplay style="display:none">
      <source src="{data_uri}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ── Page configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Object Narrator",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — high-contrast, accessible design ───────────────────────────

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Root palette ── */
:root {
  --bg:        #0a0a0f;
  --surface:   #12121a;
  --border:    #2a2a3d;
  --accent:    #00e5ff;
  --accent2:   #ff6b35;
  --success:   #00ff88;
  --warn:      #ffd60a;
  --text:      #f0f0ff;
  --muted:     #8888aa;
  --radius:    12px;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Syne', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 2px solid var(--border);
}

/* ── Headers ── */
h1 { font-size: 2.4rem !important; font-weight: 800 !important;
     background: linear-gradient(135deg, var(--accent), var(--accent2));
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     letter-spacing: -1px; }
h2, h3 { color: var(--accent) !important; font-weight: 700 !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1.1rem !important;
  padding: 0.7rem 1.6rem !important;
  border-radius: var(--radius) !important;
  border: 2px solid var(--accent) !important;
  background: transparent !important;
  color: var(--accent) !important;
  transition: all 0.2s ease !important;
  width: 100%;
}
.stButton > button:hover {
  background: var(--accent) !important;
  color: #000 !important;
  box-shadow: 0 0 20px rgba(0,229,255,0.4) !important;
}

/* ── Toggle / checkbox ── */
.stCheckbox > label { font-size: 1.05rem !important; color: var(--text) !important; }

/* ── Sliders ── */
.stSlider > label { color: var(--muted) !important; }

/* ── Info / success / warning boxes ── */
.stAlert { border-radius: var(--radius) !important; font-size: 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-size: 1.8rem !important;
  font-weight: 800 !important;
}

/* ── Detection log box ── */
.detection-log {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  max-height: 260px;
  overflow-y: auto;
  color: var(--text);
}
.detection-log .entry-obj   { color: var(--accent);  }
.detection-log .entry-ocr   { color: var(--success); }
.detection-log .entry-ts    { color: var(--muted); font-size: 0.75rem; }

/* ── OCR result box ── */
.ocr-box {
  background: #0d1a0d;
  border: 2px solid var(--success);
  border-radius: var(--radius);
  padding: 1.2rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 1rem;
  color: var(--success);
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 80px;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Camera image caption ── */
.camera-label {
  text-align: center;
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 4px;
}

/* ── Status pill ── */
.pill {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.5px;
}
.pill-on  { background: rgba(0,255,136,0.15); color: var(--success); border: 1px solid var(--success); }
.pill-off { background: rgba(136,136,170,0.1); color: var(--muted);  border: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ── Session-state defaults ───────────────────────────────────────────────────

def _init_state():
    defaults = {
        "running":          False,
        "reading_mode":     False,
        "last_spoken":      {},          # label → last spoken timestamp
        "detection_log":    [],          # list of log-entry dicts
        "ocr_text":         "",
        "total_detections": 0,
        "audio_uri":        "",
        "cooldown":         3.0,
        "conf_threshold":   0.60,
        "frame_snapshot":   None,        # numpy array for OCR
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # Start / Stop
    btn_label = "⏹ Stop Narrator" if st.session_state.running else "▶ Start Narrator"
    if st.button(btn_label, key="btn_toggle"):
        st.session_state.running = not st.session_state.running
        if not st.session_state.running:
            st.session_state.reading_mode = False

    st.markdown("---")

    # Reading Mode toggle
    reading_toggle = st.toggle(
        "📖 Reading Mode (OCR)",
        value=st.session_state.reading_mode,
        disabled=not st.session_state.running,
        help="Capture the current frame and read any text aloud.",
    )
    st.session_state.reading_mode = reading_toggle

    if st.session_state.running and reading_toggle:
        if st.button("📸 Capture & Read Frame", key="btn_ocr"):
            st.session_state["do_ocr"] = True
    else:
        st.session_state["do_ocr"] = False

    st.markdown("---")

    # Confidence threshold
    st.session_state.conf_threshold = st.slider(
        "🎯 Detection Confidence",
        min_value=0.30, max_value=0.95,
        value=st.session_state.conf_threshold,
        step=0.05,
        format="%.0f%%",
        help="Only announce objects detected above this confidence level.",
    )

    # Cooldown
    st.session_state.cooldown = st.slider(
        "⏱ Speech Cooldown (seconds)",
        min_value=1.0, max_value=10.0,
        value=st.session_state.cooldown,
        step=0.5,
        help="Minimum gap before the same object is announced again.",
    )

    st.markdown("---")

    # Clear log
    if st.button("🗑 Clear Log", key="btn_clear"):
        st.session_state.detection_log = []
        st.session_state.total_detections = 0
        st.session_state.ocr_text = ""

    st.markdown("---")

    # Status pill
    status_html = (
        '<span class="pill pill-on">● ACTIVE</span>'
        if st.session_state.running else
        '<span class="pill pill-off">○ STOPPED</span>'
    )
    st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)

# ── Main layout ──────────────────────────────────────────────────────────────

st.markdown("# 👁️ AI Object Narrator")
st.markdown(
    "Real-time object detection & reading assistant for visually impaired users. "
    "Press **▶ Start Narrator** in the sidebar to begin."
)
st.markdown("---")

col_cam, col_info = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown("### 📷 Live Camera Feed")
    cam_placeholder   = st.empty()
    status_placeholder = st.empty()

with col_info:
    st.markdown("### 📊 Session Stats")
    m1, m2 = st.columns(2)
    metric_detections = m1.empty()
    metric_objects    = m2.empty()

    st.markdown("### 🔊 Last Announcement")
    last_ann_placeholder = st.empty()

    st.markdown("### 📝 Detection Log")
    log_placeholder = st.empty()

# OCR result section (full width, below)
st.markdown("---")
st.markdown("### 📖 OCR — Extracted Text")
ocr_placeholder = st.empty()

# Hidden audio player placeholder
audio_placeholder = st.empty()

# ── Helper: draw bounding boxes ──────────────────────────────────────────────

COLORS = {
    "person": (0, 229, 255),
    "chair":  (255, 107, 53),
    "default":(255, 214, 10),
}

def draw_boxes(frame: np.ndarray, results) -> tuple[np.ndarray, list[str]]:
    """Draw YOLO detections; return annotated frame and list of label strings."""
    h, w = frame.shape[:2]
    announced = []
    threshold = st.session_state.conf_threshold

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < threshold:
            continue
        cls_id = int(box.cls[0])
        label  = results[0].names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS.get(label, COLORS["default"])

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        text  = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

        announced.append(label)

    return frame, announced

# ── Helper: cooldown-aware speech ───────────────────────────────────────────

def should_speak(label: str) -> bool:
    now = time.time()
    last = st.session_state.last_spoken.get(label, 0)
    if now - last >= st.session_state.cooldown:
        st.session_state.last_spoken[label] = now
        return True
    return False

def build_announcement(labels: list[str]) -> str:
    """Pick the most 'important' label and form a natural phrase."""
    priority = ["person", "car", "dog", "cat", "bicycle", "motorcycle"]
    for p in priority:
        if p in labels:
            return f"{p.capitalize()} detected"
    if labels:
        return f"{labels[0].capitalize()} in front"
    return ""

# ── Helper: log entry ────────────────────────────────────────────────────────

def add_log(text: str, kind: str = "obj"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.detection_log.insert(0, {"text": text, "kind": kind, "ts": ts})
    if len(st.session_state.detection_log) > 60:
        st.session_state.detection_log = st.session_state.detection_log[:60]

# ── Helper: render log HTML ──────────────────────────────────────────────────

def render_log() -> str:
    if not st.session_state.detection_log:
        return '<div class="detection-log"><span style="color:#555">No detections yet…</span></div>'
    rows = ""
    for e in st.session_state.detection_log[:30]:
        cls = "entry-ocr" if e["kind"] == "ocr" else "entry-obj"
        icon = "📖" if e["kind"] == "ocr" else "🔵"
        rows += (
            f'<div style="margin-bottom:6px">'
            f'  <span class="entry-ts">[{e["ts"]}]</span> '
            f'  {icon} <span class="{cls}">{e["text"]}</span>'
            f'</div>'
        )
    return f'<div class="detection-log">{rows}</div>'

# ── OCR logic ────────────────────────────────────────────────────────────────

def run_ocr(frame: np.ndarray) -> str:
    reader = load_ocr()
    results = reader.readtext(frame, detail=0, paragraph=True)
    return "\n".join(results).strip() if results else "(No text found)"

# ── Main camera loop ─────────────────────────────────────────────────────────

def camera_loop():
    model = load_yolo()
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        status_placeholder.error(
            "❌ Cannot access webcam. "
            "Check that your camera is connected and browser permissions are granted."
        )
        st.session_state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    unique_objects: set[str] = set()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.warning("⚠️ Frame dropped — retrying…")
            time.sleep(0.05)
            continue

        # ── OCR mode: capture snapshot on demand ─────────────────────────
        if st.session_state.get("do_ocr"):
            st.session_state.frame_snapshot = frame.copy()
            st.session_state["do_ocr"]      = False

            ocr_text = run_ocr(frame)
            st.session_state.ocr_text = ocr_text

            if ocr_text and ocr_text != "(No text found)":
                audio_uri = tts_speak(ocr_text[:400])   # cap length for TTS
                st.session_state.audio_uri = audio_uri
                add_log(f'OCR: "{ocr_text[:60]}…"' if len(ocr_text) > 60 else f'OCR: "{ocr_text}"', kind="ocr")

        # ── Object detection ─────────────────────────────────────────────
        results = model(frame, verbose=False)
        annotated, detected_labels = draw_boxes(frame, results)

        # Filter by confidence (already done inside draw_boxes) and cooldown
        to_speak = []
        for lbl in detected_labels:
            unique_objects.add(lbl)
            if should_speak(lbl):
                to_speak.append(lbl)
                st.session_state.total_detections += 1
                add_log(f"{lbl.capitalize()} detected", kind="obj")

        # Build & speak announcement
        if to_speak:
            phrase = build_announcement(to_speak)
            if phrase:
                st.session_state.audio_uri  = tts_speak(phrase)
                st.session_state.last_phrase = phrase

        # ── Update UI ────────────────────────────────────────────────────
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        cam_placeholder.image(rgb, channels="RGB", use_container_width=True)
        st.markdown('<p class="camera-label">Live — YOLOv8n</p>', unsafe_allow_html=True)

        # Stats
        metric_detections.metric("Total Announcements", st.session_state.total_detections)
        metric_objects.metric("Unique Objects",         len(unique_objects))

        # Last announcement
        phrase = getattr(st.session_state, "last_phrase", "—")
        last_ann_placeholder.markdown(
            f'<div style="font-size:1.3rem;font-weight:700;color:#00e5ff;'
            f'background:#12121a;border:1px solid #2a2a3d;border-radius:12px;'
            f'padding:0.8rem 1.2rem;">{phrase}</div>',
            unsafe_allow_html=True
        )

        # Log
        log_placeholder.markdown(render_log(), unsafe_allow_html=True)

        # OCR text
        if st.session_state.ocr_text:
            ocr_placeholder.markdown(
                f'<div class="ocr-box">{st.session_state.ocr_text}</div>',
                unsafe_allow_html=True
            )

        # Audio (injected once; browser plays it)
        if st.session_state.audio_uri:
            audio_placeholder.markdown(
                f'<audio autoplay style="display:none">'
                f'<source src="{st.session_state.audio_uri}" type="audio/mp3"></audio>',
                unsafe_allow_html=True
            )
            st.session_state.audio_uri = ""   # reset so we don't replay

        time.sleep(0.03)   # ~30 fps target

    cap.release()
    cam_placeholder.info("📷 Camera feed stopped. Press ▶ Start Narrator to resume.")

# ── Idle / stopped state UI ──────────────────────────────────────────────────

if not st.session_state.running:
    metric_detections.metric("Total Announcements", st.session_state.total_detections)
    metric_objects.metric("Unique Objects", 0)
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    if st.session_state.ocr_text:
        ocr_placeholder.markdown(
            f'<div class="ocr-box">{st.session_state.ocr_text}</div>',
            unsafe_allow_html=True
        )

    cam_placeholder.markdown("""
    <div style="
        background:#12121a;
        border:2px dashed #2a2a3d;
        border-radius:12px;
        height:380px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        color:#555;
        font-size:1.1rem;
        gap:12px;
    ">
      <span style="font-size:3.5rem">📷</span>
      <span>Camera feed will appear here</span>
      <span style="font-size:0.85rem">Press <strong style="color:#00e5ff">▶ Start Narrator</strong> in the sidebar</span>
    </div>
    """, unsafe_allow_html=True)

    last_ann_placeholder.markdown(
        '<div style="font-size:1.1rem;color:#555;background:#12121a;border:1px solid #2a2a3d;'
        'border-radius:12px;padding:0.8rem 1.2rem;">—</div>',
        unsafe_allow_html=True
    )

# ── Run camera loop when active ──────────────────────────────────────────────

if st.session_state.running:
    camera_loop()

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#444;font-size:0.8rem;">'
    'AI Object Narrator · YOLOv8 · EasyOCR · gTTS · Streamlit'
    '</p>',
    unsafe_allow_html=True
)
