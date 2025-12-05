# app.py
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import json
import os
import tempfile
from collections import Counter
import time
import base64

# ==============================================================
# 1. Page config + CSS
# ==============================================================
st.set_page_config(page_title="Driver Behavior AI", page_icon="car", layout="wide")

st.markdown(
    """
<style>
    .big-title {font-size:3.2rem; font-weight:800; color:#1E3A8A; text-align:center; margin-bottom:0.5rem;}
    .subtitle {font-size:1.4rem; color:#4B5563; text-align:center; margin-bottom:2rem;}
    .status-box {padding:1.2rem; border-radius:12px; font-weight:bold; text-align:center; margin:1rem 0; font-size:1.4rem;}
    .safe {background:#DCFCE7; color:#166534; border:2px solid #BBF7D0;}
    .danger {background:#FECACA; color:#991B1B; border:2px solid #FCA5A5;}
    .warning {background:#FEF3C7; color:#92400E; border:2px solid #FDE68A;}
    .stats-box {background:#F8FAFC; padding:1.5rem; border-radius:12px; border:1px solid #E2E8F0; margin:1rem 0;}
    .footer {text-align:center; margin-top:3rem; color:#6B7280; font-size:0.9rem;}
    .metric-card {background:white; padding:1rem; border-radius:10px; text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
    .live-status {font-size:2rem; font-weight:bold; text-align:center; padding:1rem; border-radius:12px; margin:1rem 0;}
    .stTabs {font-weight: bold;}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================
# 2. Load model
# ==============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "driver_distraction_model.keras")
json_path = os.path.join(current_dir, "class_indices.json")

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    if not os.path.exists(model_path): st.error("Model missing!"); st.stop()
    if not os.path.exists(json_path): st.error("JSON missing!"); st.stop()
    model = tf.keras.models.load_model(model_path)
    with open(json_path) as f: class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    predict_fn = tf.function(lambda x: model(x, training=False))
    return model, class_indices, idx_to_class, predict_fn

model, class_indices, idx_to_class, predict_fn = load_model()

# ==============================================================
# 3. Core functions
# ==============================================================
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_once(frame):
    inp = tf.convert_to_tensor(preprocess(frame))
    pred = predict_fn(inp)[0].numpy()
    idx = np.argmax(pred)
    cls = idx_to_class[idx]
    conf = pred[idx]
    return cls, conf

def get_label(cls, conf):
    if cls == "c6" and conf > 0.30: return "drinking", "High Risk"
    if cls in ["c1","c2","c3","c4","c9"] and conf > 0.28: return "using_phone", "High Risk"
    if cls == "c0" and conf > 0.5: return "safe_driving", "Safe"
    if cls == "c7" and conf > 0.7: return "turning", "Moderate Risk"
    if cls == "c8" and conf > 0.8: return "hair_makeup", "High Risk"
    if cls == "c5" and conf > 0.6: return "radio", "Moderate Risk"
    return "other_activity", "Unknown"

def draw_label(frame, label, risk):
    color_map = {
        "safe_driving": (0, 255, 0),
        "using_phone": (0, 0, 255),
        "drinking": (200, 0, 200),
        "hair_makeup": (255, 20, 147),
        "turning": (0, 255, 255),
        "radio": (100, 100, 255),
        "other_activity": (150, 150, 150)
    }
    color = color_map.get(label, (255, 255, 255))
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w-10, 110), color, -1)
    cv2.putText(frame, label.replace("_", " ").title(), (20, 65),
                cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 4)
    cv2.putText(frame, f"Risk: {risk}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame

# ==============================================================
# 4. Tabs
# ==============================================================
tab1, tab2, tab3 = st.tabs(["Upload Video", "Live Camera", "Upload Image"])

# ==============================================================
# TAB 1: Upload Video + 250 Frames Analysis
# ==============================================================
with tab1:
    st.markdown("<h2 class='big-title'>Upload Video</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("MP4 only", type=["mp4"], key="vid")

    if uploaded_file:
        orig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(orig_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_labeled.mp4").name

        cap = cv2.VideoCapture(orig_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        predictions = []
        frame_idx = 0
        predict_every = 4
        max_frames_to_analyze = 250

        progress_bar = st.progress(0)
        status = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frames_to_analyze: break
            frame_idx += 1

            label, risk = "safe_driving", "Safe"
            if frame_idx % predict_every == 0:
                cls, conf = predict_once(frame)
                label, risk = get_label(cls, conf)
                predictions.append((label, risk, frame_idx))

            frame = draw_label(frame, label, risk)
            out.write(frame)
            progress_bar.progress(frame_idx / min(total_frames, max_frames_to_analyze))

        cap.release()
        out.release()
        progress_bar.empty()

        # === عرض الفيديو ===
        st.video(output_path)

        # === إحصائيات + 250 فريم ===
        if predictions:
            col1, col2, col3, col4 = st.columns(4)
            labels = [p[0] for p in predictions]
            counter = Counter(labels)
            most = counter.most_common(1)[0]

            with col1:
                st.markdown(f"<div class='metric-card'><h3>Most Common</h3><h2>{most[0].replace('_',' ').title()}</h2><p>{most[1]}x</p></div>", unsafe_allow_html=True)
            with col2:
                safe = counter.get("safe_driving", 0)
                st.markdown(f"<div class='metric-card'><h3>Safe</h3><h2>{safe}x</h2><p>{safe/len(predictions)*100:.0f}%</p></div>", unsafe_allow_html=True)
            with col3:
                danger = sum(counter.get(k,0) for k in ["using_phone","drinking","hair_makeup"])
                st.markdown(f"<div class='metric-card'><h3>High Risk</h3><h2>{danger}x</h2><p>{danger/len(predictions)*100:.0f}%</p></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='metric-card'><h3>Total</h3><h2>{len(predictions)}</h2><p>Predictions</p></div>", unsafe_allow_html=True)

            # جدول 250 فريم
            st.markdown("### First 250 Frames Timeline")
            timeline = []
            for i, (label, risk, fidx) in enumerate(predictions):
                sec = fidx / fps
                timeline.append({
                    "Time": f"{sec:.1f}s",
                    "Frame": fidx,
                    "Behavior": label.replace("_", " ").title(),
                    "Risk": "Safe" if risk == "Safe" else "High Risk" if "High" in risk else "Moderate"
                })
            st.dataframe(timeline, use_container_width=True, height=400)

            st.balloons()

        os.unlink(orig_path)
        os.unlink(output_path)

# ==============================================================
# TAB 2: Live Camera
# ==============================================================
with tab2:
    st.markdown("<h2 class='big-title'>Live Camera</h2>", unsafe_allow_html=True)
    start_btn = st.button("Start Live Detection", type="primary")
    stop_btn = st.button("Stop", type="secondary")

    if start_btn:
        st.session_state.live = True
    if stop_btn:
        st.session_state.live = False

    if getattr(st.session_state, "live", False):
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        history = []

        while cap.isOpened() and st.session_state.live:
            ret, frame = cap.read()
            if not ret: break

            cls, conf = predict_once(frame)
            label, risk = get_label(cls, conf)
            history.append((label, risk))

            frame = draw_label(frame, label, risk)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            color = "safe" if risk == "Safe" else "danger"
            status_placeholder.markdown(f"<div class='live-status {color}'>LIVE: {label.replace('_',' ').title()}</div>", unsafe_allow_html=True)

        cap.release()
        st.session_state.live = False
        st.success("Live detection stopped.")

# ==============================================================
# TAB 3: Upload Image
# ==============================================================
with tab3:
    st.markdown("<h2 class='big-title'>Upload Image</h2>", unsafe_allow_html=True)
    img_file = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"], key="img")

    if img_file:
        nparr = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            cls, conf = predict_once(img)
            label, risk = get_label(cls, conf)
            img = draw_label(img, label, risk)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{label.replace('_',' ').title()} | {risk}")
            st.markdown(f"<div class='status-box {'safe' if risk=='Safe' else 'danger'}'>Result: <strong>{label.replace('_',' ').title()}</strong> | Risk: <strong>{risk}</strong></div>", unsafe_allow_html=True)

# ==============================================================
# Footer
# ==============================================================
st.markdown("""
<div class='footer'>
    <p>Driver Behavior AI | Video + Live + Image | 250 Frames Analysis | Powered by TensorFlow</p>
</div>
""", unsafe_allow_html=True)
