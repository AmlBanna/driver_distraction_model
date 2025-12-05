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
# 1. Page config + CSS (محسّنة + أيقونات + أنيميشن)
# ==============================================================
st.set_page_config(page_title="Driver Behavior AI", page_icon="car", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;600;400&display=swap');
    
    * {font-family: 'Poppins', sans-serif;}
    
    .big-title {font-size:3.5rem; font-weight:800; color:#1E3A8A; text-align:center; margin-bottom:0.5rem; 
                background: linear-gradient(90deg, #1E3A8A, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {font-size:1.5rem; color:#4B5563; text-align:center; margin-bottom:2rem; font-weight:600;}
    
    .status-box {padding:1.4rem; border-radius:16px; font-weight:bold; text-align:center; margin:1.2rem 0; font-size:1.5rem; 
                 box-shadow: 0 6px 12px rgba(0,0,0,0.1);}
    .safe {background: linear-gradient(135deg, #BBF7D0, #DCFCE7); color:#166534; border:3px solid #86EFAC;}
    .danger {background: linear-gradient(135deg, #FCA5A5, #FECACA); color:#991B1B; border:3px solid #F87171;}
    .warning {background: linear-gradient(135deg, #FDE68A, #FEF3C7); color:#92400E; border:3px solid #FBBF24;}
    
    .stats-box {background: linear-gradient(135deg, #F8FAFC, #F1F5F9); padding:1.8rem; border-radius:16px; 
                border:2px solid #E2E8F0; margin:1.5rem 0; box-shadow: 0 8px 16px rgba(0,0,0,0.08);}
    .metric-card {background: white; padding:1.4rem; border-radius:14px; text-align:center; 
                  box-shadow: 0 6px 12px rgba(0,0,0,0.1); transition: all 0.3s; border: 1px solid #E5E7EB;}
    .metric-card:hover {transform: translateY(-5px); box-shadow: 0 12px 20px rgba(0,0,0,0.15);}
    
    .live-status {font-size:2.2rem; font-weight:bold; text-align:center; padding:1.2rem; border-radius:16px; 
                  margin:1.2rem 0; box-shadow: 0 6px 12px rgba(0,0,0,0.15);}
    
    .stTabs {font-weight: bold; font-size:1.3rem;}
    .stTabs > div > div > div > div {background: #1E3A8A; color: white; border-radius: 12px; padding: 0.8rem 1.5rem;}
    
    .footer {text-align:center; margin-top:4rem; color:#6B7280; font-size:1rem; font-weight:500;}
    
    .video-container {border-radius: 16px; overflow: hidden; box-shadow: 0 10px 20px rgba(0,0,0,0.2); margin: 1.5rem 0;}
    .upload-box {border: 3px dashed #3B82F6; border-radius: 16px; padding: 2rem; text-align: center; background: #F8FAFC;}
    
    @keyframes pulse {
        0% {box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.5);}
        70% {box-shadow: 0 0 0 15px rgba(59, 130, 246, 0);}
        100% {box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);}
    }
    .pulse {animation: pulse 2s infinite;}
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
    cv2.rectangle(frame, (10, 10), (w-10, 120), color, -1)
    cv2.putText(frame, label.replace("_", " ").title(), (25, 70),
                cv2.FONT_HERSHEY_DUPLEX, 2.8, (255, 255, 255), 5)
    cv2.putText(frame, f"Risk: {risk}", (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
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
    st.markdown("<p class='subtitle'>AI will analyze first 250 frames and show labels on video</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("**Drop your MP4 video here**", type=["mp4"], key="vid", help="Max 200MB")

    if uploaded_file:
        # Save & process
        orig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(orig_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_ai.mp4").name

        cap = cv2.VideoCapture(orig_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        predictions = []
        frame_idx = 0
        predict_every = 4
        max_frames_to_analyze = 250

        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.markdown("**Analyzing video...**")

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
        status_text.success("Analysis Complete!")

        # === عرض الفيديو بالتنبؤ ===
        st.markdown("## Analyzed Video")
        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
        st.video(output_path)
        st.markdown("</div>", unsafe_allow_html=True)

        # === إحصائيات + 250 فريم ===
        if predictions:
            col1, col2, col3, col4 = st.columns(4)
            labels = [p[0] for p in predictions]
            counter = Counter(labels)
            most = counter.most_common(1)[0]

            with col1:
                st.markdown(f"<div class='metric-card pulse'><h3>Most Common</h3><h2>{most[0].replace('_',' ').title()}</h2><p>{most[1]}x</p></div>", unsafe_allow_html=True)
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
            for i, (label, risk, fidx) in enumerate(predictions):
                sec = fidx / fps
                timeline.append({
                    "Time": f"{sec:.1f}s",
                    "Frame": fidx,
                    "Behavior": label.replace("_", " ").title(),
                    "Risk": "Safe" if risk == "Safe" else "High Risk" if "High" in risk else "Moderate"
                })
            st.dataframe(timeline, use_container_width=True, height=450)

            st.balloons()

        os.unlink(orig_path)
        os.unlink(output_path)

# ==============================================================
# TAB 2: Live Camera
# ==============================================================
with tab2:
    st.markdown("<h2 class='big-title'>Live Camera</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Real-time AI detection from your webcam</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("Start Live", type="primary", use_column_width=True)
        stop_btn = st.button("Stop", type="secondary", use_column_width=True)
    
    if start_btn:
        st.session_state.live = True
    if stop_btn:
        st.session_state.live = False

    if getattr(st.session_state, "live", False):
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Please check permissions.")
            st.stop()

        while cap.isOpened() and st.session_state.live:
            ret, frame = cap.read()
            if not ret: break

            cls, conf = predict_once(frame)
            label, risk = get_label(cls, conf)
            frame = draw_label(frame, label, risk)
            
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            color_class = "safe" if risk == "Safe" else "danger"
            status_placeholder.markdown(f"<div class='live-status {color_class}'>LIVE: {label.replace('_',' ').title()}</div>", unsafe_allow_html=True)

        cap.release()
        st.session_state.live = False
        st.success("Live detection stopped.")

# ==============================================================
# TAB 3: Upload Image
# ==============================================================
with tab3:
    st.markdown("<h2 class='big-title'>Upload Image</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Instant AI analysis of a single frame</p>", unsafe_allow_html=True)
    
    img_file = st.file_uploader("**Drop image here**", type=["jpg", "jpeg", "png"], key="img")

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
    <p>Driver Behavior AI | Video + Live + Image | 250 Frames Analysis | Powered by <strong>TensorFlow</strong> & <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)

st.balloons()
