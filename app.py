# app.py
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import json
import os
from collections import Counter
import tempfile
import time
import base64
from gtts import gTTS

# ==============================================================
# 1. Page config + CSS
# ==============================================================
st.set_page_config(page_title="Driver Behavior AI", page_icon="car", layout="centered")

st.markdown(
    """
<style>
    .big-title {font-size:3rem; font-weight:800; color:#1E3A8A; text-align:center;}
    .subtitle {font-size:1.3rem; color:#4B5563; text-align:center; margin-bottom:2rem;}
    .status-card {padding:1rem; border-radius:12px; font-weight:bold; text-align:center; margin:1rem 0;}
    .safe {background:#DCFCE7; color:#166534; border:2px solid #BBF7D0;}
    .danger {background:#FECACA; color:#991B1B; border:2px solid #FCA5A5;}
    .warning {background:#FEF3C7; color:#92400E; border:2px solid #FDE68A;}
    .real-time {font-size:1.6rem; font-weight:bold; text-align:center; padding:0.6rem; border-radius:10px; margin:0.8rem 0;}
    .stButton>button {background:#1E3A8A; color:white; border-radius:8px; padding:0.6rem 1.2rem;}
    .footer {text-align:center; margin-top:3rem; color:#6B7280; font-size:0.9rem;}
    .stats-box {background:#F8FAFC; padding:1.2rem; border-radius:12px; border:1px solid #E2E8F0;}
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

@st.cache_resource(show_spinner="Loading AI model…")
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
# 3. Preprocess + Predict
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

# ==============================================================
# 4. UI
# ==============================================================
st.markdown("<h1 class='big-title'>Driver Behavior AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smooth 43-sec video • Accurate predictions every 3-5 frames</p>", unsafe_allow_html=True)

tab_vid = st.tabs(["Upload Video"])[0]

with tab_vid:
    st.write("#### Upload your 43-second video")
    uploaded_file = st.file_uploader("MP4 only", type=["mp4"], key="vid")

    if uploaded_file:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tpath = tfile.name

        # Open video
        cap = cv2.VideoCapture(tpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        st.write(f"**Duration:** {duration:.1f}s | **FPS:** {fps:.1f}")

        # UI placeholders
        video_ph = st.empty()
        status_ph = st.empty()
        progress = st.progress(0)
        stats_ph = st.empty()

        # Prediction control
        predict_every_n_frames = 3  # تنبؤ كل 3 فريمات
        frame_idx = 0
        history = []
        current_label = "safe_driving"
        current_risk = "Safe"

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            current_time = frame_idx / fps

            # تنبؤ كل N فريمات فقط
            if frame_idx % predict_every_n_frames == 0:
                cls, conf = predict_once(frame)
                label, risk = get_label(cls, conf)
                history.append((label, risk))
                current_label, current_risk = label, risk

            # رسم النص على الفريم
            color = (0, 255, 0) if current_label == "safe_driving" else (0, 0, 255)
            if "drinking" in current_label: color = (200, 0, 200)
            if "hair_makeup" in current_label: color = (255, 20, 147)

            cv2.putText(frame, current_label.replace("_", " ").title(),
                        (15, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 4)

            # عرض الفيديو + الحالة
            video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            status_color = "safe" if current_risk == "Safe" else "danger"
            status_ph.markdown(f"<div class='real-time {status_color}'>Current: {current_label.replace('_', ' ').title()}</div>", unsafe_allow_html=True)

            # تحديث التقدم
            progress.progress(frame_idx / total_frames)

            # تأخير بسيط لمحاكاة الوقت الحقيقي (اختياري)
            elapsed = time.time() - start_time
            expected = current_time
            if elapsed < expected:
                time.sleep(expected - elapsed)

        cap.release()
        os.unlink(tpath)

        # === الإحصائيات النهائية ===
        if history:
            behaviors = [h[0] for h in history]
            most_common = Counter(behaviors).most_common(1)[0]
            st.markdown("### Final Report")
            st.markdown(f"<div class='stats-box'>"
                        f"<strong>Most frequent behavior:</strong> {most_common[0].replace('_',' ').title()}<br>"
                        f"<strong>Detected in:</strong> {most_common[1]} predictions ({most_common[1]*predict_every_n_frames} frames approx.)"
                        f"</div>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.info("No predictions made.")

# ==============================================================
# Footer
# ==============================================================
st.markdown("<div class='footer'><p>Smooth • Accurate • Real-time AI</p></div>", unsafe_allow_html=True)
