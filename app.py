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

# ==============================================================
# 1. Page config + ألوان هادية وبسيطة + إيموجيز
# ==============================================================
st.set_page_config(page_title="Driver Behavior AI", page_icon="blossom", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {font-family: 'Inter', sans-serif;}
    
    .big-title {
        font-size:3.2rem; font-weight:700; 
        color:#1E293B; text-align:center; margin-bottom:0.5rem;
    }
    .subtitle {
        font-size:1.4rem; color:#64748B; text-align:center; margin-bottom:2rem; font-weight:400;
    }
    
    .status-box {
        padding:1.3rem; border-radius:16px; font-weight:600; text-align:center; 
        margin:1rem 0; font-size:1.4rem; 
        background:#F8FAFC; border:2px solid #E2E8F0;
    }
    .safe {background:#F0FDF4; color:#166534; border:2px solid #BBF7D0;}
    .danger {background:#FEF2F2; color:#991B1B; border:2px solid #FCA5A5;}
    .warning {background:#FFFBEB; color:#92400E; border:2px solid #FDE68A;}
    
    .stats-box {
        background:#FAFAFA; padding:1.6rem; border-radius:16px; 
        border:1px solid #E5E7EB; margin:1.5rem 0; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white; padding:1.3rem; border-radius:14px; text-align:center; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.06); 
        border: 1px solid #F1F5F9; transition: all 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-4px); box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .live-status {
        font-size:2.1rem; font-weight:700; text-align:center; padding:1.2rem; 
        border-radius:16px; margin:1.2rem 0; 
        background:#F1F5F9; border:2px solid #CBD5E1;
    }
    
    .stTabs > div > div > div > div {
        background: #F8FAFC; color: #1E293B; border-radius: 12px; 
        padding: 0.8rem 1.6rem; font-weight:600; border:1px solid #E2E8F0;
    }
    
    .footer {
        text-align:center; margin-top:4rem; color:#94A3B8; font-size:0.95rem; font-weight:400;
    }
    
    .video-container {
        border-radius: 18px; overflow: hidden; 
        box-shadow: 0 6px 16px rgba(0,0,0,0.08); margin: 1.6rem 0;
        border: 1px solid #E2E8F0;
    }
    
    .upload-box {
        border: 2px dashed #94A3B8; border-radius: 16px; padding: 2rem; 
        text-align: center; background: #F8FAFC; font-weight:500; color:#64748B;
    }
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

@st.cache_resource(show_spinner="Loading model...")
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
        "safe_driving": (34, 197, 94),      # Green
        "using_phone": (239, 68, 68),       # Red
        "drinking": (168, 85, 247),         # Purple
        "hair_makeup": (236, 72, 153),      # Pink
        "turning": (251, 146, 60),          # Orange
        "radio": (59, 130, 246),            # Blue
        "other_activity": (148, 163, 184)   # Gray
    }
    color = color_map.get(label, (255, 255, 255))
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w-10, 120), color, -1)
    cv2.putText(frame, f"{label.replace('_', ' ').title()}", (25, 70),
                cv2.FONT_HERSHEY_DUPLEX, 2.6, (255, 255, 255), 5)
    cv2.putText(frame, f"Risk: {risk}", (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame

# ==============================================================
# 4. Tabs
# ==============================================================
tab1, tab2, tab3 = st.tabs(["Upload Video", "Live Camera", "Upload Image"])

# ==============================================================
# TAB 1: Upload Video + 250 Frames (فيديو شغال 100%)
# ==============================================================
with tab1:
    st.markdown("<h2 class='big-title'>Upload Video</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI analyzes first 250 frames with clear labels</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drop your MP4 video here", type=["mp4"], key="vid")

    if uploaded_file:
        # حفظ الملف مؤقتًا
        input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_output.mp4").name

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        predictions = []
        frame_idx = 0
        predict_every = 4
        max_frames = 250

        progress = st.progress(0)
        status = st.empty()
        status.markdown("Analyzing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frames: break
            frame_idx += 1

            label, risk = "safe_driving", "Safe"
            if frame_idx % predict_every == 0:
                cls, conf = predict_once(frame)
                label, risk = get_label(cls, conf)
                predictions.append((label, risk, frame_idx))

            frame = draw_label(frame, label, risk)
            out.write(frame)
            progress.progress(frame_idx / min(total_frames, max_frames))

        cap.release()
        out.release()
        progress.empty()
        status.success("Analysis complete!")

        # عرض الفيديو (شغال 100%)
        st.markdown("## Analyzed Video")
        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
        st.video(output_path)
        st.markdown("</div>", unsafe_allow_html=True)

        # إحصائيات + جدول
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

            st.markdown("### First 250 Frames Timeline")
            timeline = []
            for label, risk, fidx in predictions:
                sec = fidx / fps
                timeline.append({
                    "Time": f"{sec:.1f}s",
                    "Frame": fidx,
                    "Behavior": label.replace("_", " ").title(),
                    "Risk": "Safe" if risk == "Safe" else "High Risk" if "High" in risk else "Moderate"
                })
            st.dataframe(timeline, use_container_width=True, height=450)

            st.balloons()

        # تنظيف
        os.unlink(input_path)
        os.unlink(output_path)

# ==============================================================
# TAB 2: Live Camera
# ==============================================================
with tab2:
    st.markdown("<h2 class='big-title'>Live Camera</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Real-time detection from your webcam</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("Start Live", type="primary")
    with col2:
        stop = st.button("Stop", type="secondary")

    if start: st.session_state.live = True
    if stop: st.session_state.live = False

    if getattr(st.session_state, "live", False):
        frame_ph = st.empty()
        status_ph = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access camera.")
            st.stop()

        while cap.isOpened() and st.session_state.live:
            ret, frame = cap.read()
            if not ret: break
            cls, conf = predict_once(frame)
            label, risk = get_label(cls, conf)
            frame = draw_label(frame, label, risk)
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            status_ph.markdown(f"<div class='live-status {'safe' if risk=='Safe' else 'danger'}'>LIVE: {label.replace('_',' ').title()}</div>", unsafe_allow_html=True)

        cap.release()
        st.session_state.live = False
        st.success("Live stopped.")

# ==============================================================
# TAB 3: Upload Image
# ==============================================================
with tab3:
    st.markdown("<h2 class='big-title'>Upload Image</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Instant analysis of a single frame</p>", unsafe_allow_html=True)
    
    img_file = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"], key="img")
    if img_file:
        nparr = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            cls, conf = predict_once(img)
            label, risk = get_label(cls, conf)
            img = draw_label(img, label, risk)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown(f"<div class='status-box {'safe' if risk=='Safe' else 'danger'}'>Result: <strong>{label.replace('_',' ').title()}</strong> | Risk: <strong>{risk}</strong></div>", unsafe_allow_html=True)

# ==============================================================
# Footer
# ==============================================================
st.markdown("""
<div class='footer'>
    <p>Driver Behavior AI | Simple • Calm • Smart | Powered by <strong>TensorFlow</strong> & <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)

st.balloons()
