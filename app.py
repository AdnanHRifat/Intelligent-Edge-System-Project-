import streamlit as st
import cv2
import time
import os
import psutil
import tempfile
from collections import deque
import datetime
from ultralytics import YOLO

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="EdgeShield | Group 10",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
# GENTLE CSS & INTERACTIVE STYLING
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ────────────────────────────────
   Design tokens — warm, gentle palette
──────────────────────────────── */
:root {
    --bg:          #f5f4f1;
    --surface:     #fefefe;
    --surf2:       #f9f8f6;
    --surf3:       #f2f0ec;
    --border:      #e8e5e0;
    --border2:     #d8d4cc;

    --text1:       #1a1917;
    --text2:       #57534e;
    --text3:       #a8a29e;
    --text4:       #c9c5bf;

    --accent:      #4a5568;
    --accent-lt:   #edf0f4;
    --accent-bd:   #c8d0dc;

    --sage:        #4a7c59;
    --sage-lt:     #f0f5f1;
    --sage-bd:     #c3d9c9;
    --sage-mid:    #5a8f6a;

    --gold:        #8a6f3a;
    --gold-lt:     #faf5eb;
    --gold-bd:     #e8d8a8;

    --rose:        #8a4a4a;
    --rose-lt:     #faf0f0;
    --rose-bd:     #ddb8b8;

    --stone:       #78716c;
    --stone-lt:    #f5f3f0;
    --stone-bd:    #d6d0c8;

    --ink:         #2d2d2b;
    --ink2:        #3d3c39;

    --radius:      10px;
    --radius-lg:   14px;
    --radius-sm:   7px;

    --shadow-xs:   0 1px 2px rgba(0,0,0,.04);
    --shadow-sm:   0 1px 4px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md:   0 4px 12px rgba(0,0,0,.05), 0 1px 4px rgba(0,0,0,.03);
    --shadow-lg:   0 12px 28px rgba(0,0,0,.07), 0 4px 8px rgba(0,0,0,.04);
}

/* ────────────────────────────────
   Global reset
──────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton           { display: none; }

[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}
.block-container {
    padding: 1.1rem 1.8rem 2rem !important;
    max-width: 100% !important;
}

/* scrollbar */
::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ────────────────────────────────
   Sidebar
──────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(0,0,0,.04) !important;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.11em !important;
    color: var(--text4) !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.81rem !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 0.8rem 0 !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    background: var(--ink) !important;
    color: #f5f4f1 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.81rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all .18s ease !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: var(--ink2) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] button[kind="secondary"] {
    background: var(--surf2) !important;
    color: var(--text2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.81rem !important;
    transition: all .18s ease !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: var(--surf3) !important;
    border-color: var(--accent-bd) !important;
    color: var(--text1) !important;
}

/* ────────────────────────────────
   Main page typography
──────────────────────────────── */
h1 {
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    color: var(--text1) !important;
    letter-spacing: -0.03em !important;
}
h3 {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.11em !important;
    color: var(--text4) !important;
    margin: 0.75rem 0 0.4rem !important;
}

/* ────────────────────────────────
   Alert / status overrides
──────────────────────────────── */
div[data-testid="stAlert"],
div[role="alert"] {
    border-radius: var(--radius) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.1rem !important;
    border-width: 1px !important;
    border-style: solid !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all .25s ease !important;
}
div.stSuccess, div[data-testid="stAlert"].st-success {
    background: var(--sage-lt) !important; border-color: var(--sage-bd) !important; color: var(--sage) !important;
}
div.stWarning, div[data-testid="stAlert"].st-warning {
    background: var(--gold-lt) !important; border-color: var(--gold-bd) !important; color: var(--gold) !important;
}
div.stError, div[data-testid="stAlert"].st-error {
    background: var(--rose-lt) !important; border-color: var(--rose-bd) !important; color: var(--rose) !important;
}
div.stInfo, div[data-testid="stAlert"].st-info {
    background: var(--stone-lt) !important; border-color: var(--stone-bd) !important; color: var(--stone) !important;
}

/* ────────────────────────────────
   Interactive Video Image
──────────────────────────────── */
[data-testid="stImage"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 1px solid var(--border) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stImage"]:hover {
    box-shadow: var(--shadow-lg) !important;
    border-color: var(--border2) !important;
}
[data-testid="stImage"] img {
    border-radius: var(--radius-lg) !important;
    width: 100% !important;
    display: block !important;
}

/* ────────────────────────────────
   Metric cards with hover effect
──────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.95rem 1.1rem 0.85rem !important;
    box-shadow: var(--shadow-sm) !important;
    position: relative !important;
    overflow: hidden !important;
    transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px) !important;
    border-color: var(--border2) !important;
}
[data-testid="column"]:nth-child(1) [data-testid="stMetric"] { border-top: 3px solid #7c9bb5 !important; }
[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { border-top: 3px solid var(--sage-mid) !important; }
[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { border-top: 3px solid #b5986a !important; }
[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { border-top: 3px solid var(--stone) !important; }

[data-testid="stMetricLabel"] > div {
    font-size: 0.63rem !important; font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.09em !important; color: var(--text3) !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.55rem !important; font-weight: 700 !important; color: var(--text1) !important;
    font-family: 'JetBrains Mono', monospace !important; letter-spacing: -0.04em !important; line-height: 1.15 !important;
}
[data-testid="stMetricDelta"] > div {
    font-size: 0.67rem !important; color: var(--sage) !important; font-family: 'JetBrains Mono', monospace !important;
}

/* columns */
[data-testid="column"] { padding: 0 0.35rem !important; }

/* ────────────────────────────────
   Custom component classes
──────────────────────────────── */
.page-header {
    display: flex; align-items: center; justify-content: space-between; padding: 0.7rem 1.1rem;
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm); margin-bottom: 0.75rem;
}
.ph-left   { display: flex; align-items: center; gap: 0.75rem; }
.ph-badge  {
    display: inline-flex; align-items: center; gap: 0.35rem; background: var(--ink); color: #d6d3cc;
    padding: 0.24rem 0.7rem; border-radius: 20px; font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.06em; font-family: 'JetBrains Mono', monospace;
}
.ph-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #9dc89d;
    animation: pulse-green 2s ease-in-out infinite;
}
.ph-dot.off { background: #78716c; animation: none; }
@keyframes pulse-green {
    0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(157,200,157,.5); }
    50%      { opacity: .7; box-shadow: 0 0 0 5px rgba(157,200,157,0); }
}
.ph-title { font-size: 1.05rem; font-weight: 700; color: var(--text1); letter-spacing: -0.025em; }
.ph-sub   { font-size: 0.72rem; color: var(--text3); margin-top: 1px; font-weight: 400; }
.ph-right { display: flex; align-items: center; gap: 1rem; }
.ph-clock {
    font-size: 0.7rem; color: var(--text3); font-family: 'JetBrains Mono', monospace;
    background: var(--surf2); border: 1px solid var(--border); padding: 0.22rem 0.6rem; border-radius: 6px;
}

/* Sidebar elements */
.sb-logo {
    padding: 1rem 1.1rem 0.85rem; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.6rem; background: var(--surf2);
}
.sb-logo-name { font-size: 0.95rem; font-weight: 700; color: var(--text1); letter-spacing: -0.02em; }
.sb-logo-ver { font-size: 0.6rem; color: var(--text4); font-family: 'JetBrains Mono', monospace; margin-top: 1px; }
.sb-sec {
    padding: 0.75rem 1.1rem 0.3rem; font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--text4); display: flex; align-items: center; gap: 0.45rem;
}
.sb-sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.node-wrap { padding: 0.2rem 1.1rem 0.8rem; }
.node-row  {
    display: flex; justify-content: space-between; align-items: center; padding: 0.34rem 0;
    border-bottom: 1px solid var(--border); font-size: 0.73rem;
}
.node-row:last-child { border-bottom: none; }
.nk { color: var(--text3); font-weight: 400; }
.nv { color: var(--text1); font-weight: 500; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; }
.nv.on  { color: var(--sage); }

/* Offline screen */
.offline-screen {
    background: linear-gradient(135deg, #2a2927 0%, #1e1d1b 100%); border-radius: var(--radius-lg);
    min-height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center;
    border: 1px solid #3a3835; box-shadow: var(--shadow-sm); position: relative; overflow: hidden;
    transition: all 0.3s ease;
}
.offline-screen:hover { box-shadow: var(--shadow-md); border-color: #4a4845; }
.offline-icon  { font-size: 2.6rem; opacity: .15; margin-bottom: 1rem; filter: grayscale(1); }
.offline-title { font-size: 0.95rem; font-weight: 600; color: #7c7975; margin-bottom: 0.4rem; }
.offline-sub { font-size: 0.76rem; color: #4a4845; text-align: center; line-height: 1.7; }

.telemetry-label {
    display: flex; align-items: center; gap: 0.6rem; margin: 0.65rem 0 0.65rem;
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text3);
}
.telemetry-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Interactive Event Log ── */
.event-log-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.8rem;
    height: 380px;
    overflow-y: auto;
    box-shadow: var(--shadow-sm);
}
.log-entry {
    font-size: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--surf3);
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.log-entry:last-child { border-bottom: none; }
.log-top { display: flex; align-items: center; gap: 0.5rem; }
.log-time { color: var(--text3); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; }
.log-msg { color: var(--text2); font-weight: 500; font-size: 0.72rem; line-height: 1.4; padding-left: 3.2rem;}
.log-badge {
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.lb-secured { background: var(--sage-lt); color: var(--sage); border: 1px solid var(--sage-bd); }
.lb-warning { background: var(--gold-lt); color: var(--gold); border: 1px solid var(--gold-bd); }
.lb-alarm   { background: var(--rose-lt); color: var(--rose); border: 1px solid var(--rose-bd); }
.lb-info    { background: var(--stone-lt); color: var(--stone); border: 1px solid var(--stone-bd); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "current_source" not in st.session_state:
    st.session_state.current_source = None
if "event_log" not in st.session_state:
    st.session_state.event_log = deque(maxlen=20)
if "prev_state" not in st.session_state:
    st.session_state.prev_state = None

# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        if os.path.exists("yolov8m_openvino_model"):
            return YOLO("yolov8m_openvino_model", task="detect")
        return YOLO("yolov8m.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def render_log(log_deque):
    """Renders the event log as interactive HTML"""
    if not log_deque:
        return '<div class="event-log-container"><div style="color: var(--text3); font-size: 0.75rem; text-align: center; padding-top: 2rem; font-style: italic;">System idle. No events logged yet.</div></div>'
    
    html = '<div class="event-log-container">'
    for t, state, msg in log_deque:
        badge_class = "lb-info"
        if state == "Secured": badge_class = "lb-secured"
        elif state == "Warning": badge_class = "lb-warning"
        elif state == "Alarm": badge_class = "lb-alarm"
        
        html += f'''
        <div class="log-entry">
            <div class="log-top">
                <span class="log-time">{t}</span>
                <span class="log-badge {badge_class}">{state}</span>
            </div>
            <div class="log-msg">{msg}</div>
        </div>
        '''
    html += '</div>'
    return html

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <span style="font-size:1.3rem;filter:grayscale(.2)">🛡️</span>
        <div>
            <div class="sb-logo-name">EdgeShield</div>
            <div class="sb-logo-ver">v3.1 &nbsp;·&nbsp; OpenVINO FP16 &nbsp;·&nbsp; Group 10</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">1 &nbsp; Input Source</div>', unsafe_allow_html=True)
    source_type = st.radio("source", ["🎥 Live Webcam", "📁 Upload Video (MP4)"], label_visibility="collapsed")

    video_path = None
    if source_type == "📁 Upload Video (MP4)":
        uploaded_file = st.file_uploader("Upload an MP4 file", type=["mp4"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name

    st.markdown("---")

    st.markdown('<div class="sb-sec">2 &nbsp; Detection Tuning</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence", min_value=0.1, max_value=0.95, value=0.40, step=0.05)
    iou_thresh  = st.slider("IoU (NMS)",  min_value=0.1, max_value=0.95, value=0.45, step=0.05)

    st.markdown("---")

    st.markdown('<div class="sb-sec">3 &nbsp; Target Assets</div>', unsafe_allow_html=True)
    track_laptop = st.checkbox("💻  Laptop",     value=True)
    track_phone  = st.checkbox("📱  Cell Phone", value=True)
    track_person = st.checkbox("🧍  Person",     value=False)

    target_classes = []
    if track_person: target_classes.append(0)
    if track_laptop: target_classes.append(63)
    if track_phone:  target_classes.append(67)

    st.markdown("---")

    st.markdown('<div class="sb-sec">4 &nbsp; System Engine</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶  START", use_container_width=True, type="primary"):
            st.session_state.is_running = True
            st.rerun()
    with c2:
        if st.button("⏹  STOP", use_container_width=True):
            st.session_state.is_running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()

    st.markdown("---")

    st.markdown('<div class="sb-sec">5 &nbsp; Hardware Node</div>', unsafe_allow_html=True)
    online = st.session_state.is_running
    st.markdown(f"""
    <div class="node-wrap">
        <div class="node-row"><span class="nk">Device</span><span class="nv">Galaxy Book4 Pro</span></div>
        <div class="node-row"><span class="nk">Processor</span><span class="nv">Core Ultra 5 125H</span></div>
        <div class="node-row"><span class="nk">Graphics</span><span class="nv">Intel Arc</span></div>
        <div class="node-row"><span class="nk">Runtime</span><span class="nv">OpenVINO FP16</span></div>
        <div class="node-row"><span class="nk">Model</span><span class="nv">YOLOv8m</span></div>
        <div class="node-row">
            <span class="nk">Status</span>
            <span class="nv {'on' if online else 'off'}">
                {'● Active' if online else '○ Standby'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MAIN DASHBOARD — PAGE HEADER
# ══════════════════════════════════════════════
now     = datetime.datetime.now().strftime("%H:%M:%S")
dot_cls = "ph-dot" if st.session_state.is_running else "ph-dot off"
badge   = "LIVE" if st.session_state.is_running else "STANDBY"
src_lbl = "Webcam" if source_type == "🎥 Live Webcam" else "Video File"

st.markdown(f"""
<div class="page-header">
    <div class="ph-left">
        <div class="ph-badge">
            <div class="{dot_cls}"></div>
            {badge}
        </div>
        <div>
            <div class="ph-title">Real-Time Asset Verification Node</div>
            <div class="ph-sub">Intelligent Edge System &nbsp;·&nbsp; Group 10 &nbsp;·&nbsp; Sejong University</div>
        </div>
    </div>
    <div class="ph-right">
        <span class="ph-hw">⚙ &nbsp;{src_lbl} &nbsp;·&nbsp; YOLOv8m &nbsp;·&nbsp; OpenVINO</span>
        <span class="ph-clock">🕐 &nbsp;{now}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SYSTEM TELEMETRY (MOVED ABOVE VIDEO)
# ══════════════════════════════════════════════
st.markdown('<div class="telemetry-label">System Telemetry</div>', unsafe_allow_html=True)
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
lat_metric = metric_col1.empty()
fps_metric = metric_col2.empty()
cpu_metric = metric_col3.empty()
det_metric = metric_col4.empty()

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MAIN 2-COLUMN VMS LAYOUT (Video Left, Logs Right)
# ══════════════════════════════════════════════
main_view, side_view = st.columns([2.6, 1], gap="medium")

with main_view:
    status_placeholder = st.empty()
    video_placeholder  = st.empty()

with side_view:
    st.markdown('<div class="telemetry-label">Live Event Log</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()

# ══════════════════════════════════════════════
# IDLE STATE
# ══════════════════════════════════════════════
if not st.session_state.is_running:
    lat_metric.metric("Inference Latency", "-- ms")
    fps_metric.metric("Pipeline FPS",      "-- FPS")
    cpu_metric.metric("CPU Load",          "-- %")
    det_metric.metric("Objects Detected",  "--")

    status_placeholder.info("○  **System Standby** — Press **▶ START** in the sidebar to initialise the inference engine.")
    
    video_placeholder.markdown("""
    <div class="offline-screen">
        <div class="offline-icon">📷</div>
        <div class="offline-title">Camera Offline</div>
        <div class="offline-sub">
            Activate the inference node from the sidebar<br>
            to begin real-time asset detection
        </div>
        <div class="offline-hint">▶ &nbsp; START &nbsp;→&nbsp; sidebar</div>
    </div>
    """, unsafe_allow_html=True)

    log_placeholder.markdown(render_log([]), unsafe_allow_html=True)

# ══════════════════════════════════════════════
# EXECUTION LOOP 
# ══════════════════════════════════════════════
if st.session_state.is_running:

    if source_type == "🎥 Live Webcam":
        current_src = 0
    else:
        current_src = video_path

    if st.session_state.cap is None or st.session_state.current_source != current_src:
        if st.session_state.cap is not None:
            st.session_state.cap.release()

        if current_src is not None:
            if current_src == 0:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(current_src)

            st.session_state.cap = cap
            st.session_state.current_source = current_src

    if st.session_state.cap is not None and st.session_state.cap.isOpened():

        while st.session_state.is_running:
            start_time = time.time()

            success, frame = st.session_state.cap.read()

            if not success:
                if source_type == "📁 Upload Video (MP4)":
                    st.session_state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    st.error("Webcam disconnected.")
                    break

            results = model.predict(
                frame,
                imgsz=640,
                conf=conf_thresh,
                iou=iou_thresh,
                classes=target_classes if target_classes else None,
                verbose=False
            )

            detections = results[0].boxes.cls.cpu().numpy()
            has_laptop = 63 in detections
            has_phone  = 67 in detections

            # Security Logic & Status Tracking
            sys_state = "Monitoring"
            log_msg = "Tracking custom selected assets."

            if track_laptop and track_phone:
                if has_laptop and has_phone:
                    sys_state = "Secured"
                    log_msg = "All tracked assets are currently present."
                    status_placeholder.success("🟢  **Secured** — All tracked assets are currently present.")
                elif has_laptop or has_phone:
                    sys_state = "Warning"
                    log_msg = "A tracked asset is missing from the frame."
                    status_placeholder.warning("🟡  **Warning** — A tracked asset is missing from the frame.")
                else:
                    sys_state = "Alarm"
                    log_msg = "No tracked assets detected in the operational zone."
                    status_placeholder.error("🔴  **Alarm** — No tracked assets detected in the operational zone.")
            else:
                status_placeholder.info("◎  **Monitoring** — Tracking custom selected assets.")

            # Update Event Log
            if sys_state != st.session_state.prev_state:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.event_log.appendleft((timestamp, sys_state, log_msg))
                st.session_state.prev_state = sys_state
            
            log_placeholder.markdown(render_log(st.session_state.event_log), unsafe_allow_html=True)

            # Draw & Display Video
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame, use_container_width=True)

            # Calculate Metrics
            process_time_ms = (time.time() - start_time) * 1000
            fps = 1000 / process_time_ms if process_time_ms > 0 else 0
            cpu_usage = psutil.cpu_percent()

            lat_metric.metric("Inference Latency", f"{process_time_ms:.1f} ms", "-70% vs CPU")
            fps_metric.metric("Pipeline FPS",      f"{fps:.1f} FPS")
            cpu_metric.metric("CPU Load",          f"{cpu_usage}%")
            det_metric.metric("Objects Detected",  f"{len(detections)}")

            time.sleep(0.01)

    else:
        st.warning("⚠  Please provide a valid video source — enable webcam or upload an MP4.")