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
# 1. Page config + CSS
# ==============================================================
st.set_page_config(page_title="Driver Behavior AI", page_icon="car", layout="centered")

st.markdown(
    """
<style>
    .big-title {font-size:3rem; font-weight:800; color:#1E3A8A; text-align:center;}
    .subtitle {font-size:1.3rem; color:#4B5563; text-align:center; margin-bottom:1rem;}
    .status-box {padding:1rem; border-radius:12px; font-weight:bold; text-align:center; margin:1rem 0; font-size:1.3rem;}
    .safe {background:#DCFCE7; color:#166534; border:2px solid #BBF7D0;}
    .danger {background:#FECACA; color:#991B1B; border:2px solid #FCA5A5;}
    .warning {background:#FEF3C7; color:#92400E; border:2px solid #FDE68A;}
    .stats-box {background:#F8FAFC; padding:1.5rem; border-radius:12px; border:1px solid #E2E8F0; margin-top:1rem;}
    .footer {text-align:center; margin-top:3rem; color:#6B7280; font-size:0.9rem;}
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
st.markdown("<p class='subtitle'>Full 43-sec video • Smooth playback • Real-time AI analysis</p>", unsafe_allow_html=True)

st.write("#### Upload your video (MP4 only)")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # --- عرض الفيديو أولاً (يتحرك طبيعي) ---
    st.video(uploaded_file)

    # --- تحليل في الخلفية ---
    with st.spinner("Analyzing video in background..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tpath = tfile.name

        cap = cv2.VideoCapture(tpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        st.write(f"**Duration:** {duration:.1f}s | **Frames:** {total_frames}")

        # تحليل كل 5 فريمات
        predict_every = 5
        frame_idx = 0
        predictions = []
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            if frame_idx % predict_every == 0:
                cls, conf = predict_once(frame)
                label, risk = get_label(cls, conf)
                predictions.append((label, risk, frame_idx))

            progress_bar.progress(frame_idx / total_frames)

        cap.release()
        os.unlink(tpath)
        progress_bar.empty()

    # --- نتائج التحليل ---
    if predictions:
        # إحصائيات
        labels = [p[0] for p in predictions]
        most_common = Counter(labels).most_common(1)[0]

        st.markdown("### Analysis Complete!")
        st.markdown(f"<div class='stats-box'>"
                    f"<strong>Most frequent behavior:</strong> <span style='color:#1E3A8A; font-weight:bold;'>{most_common[0].replace('_',' ').title()}</span><br>"
                    f"<strong>Detected in:</strong> {most_common[1]} samples (~{most_common[1]*predict_every} frames)<br>"
                    f"<strong>Total predictions:</strong> {len(predictions)}"
                    f"</div>", unsafe_allow_html=True)

        # جدول النتائج
        st.markdown("#### Prediction Timeline")
        timeline = ""
        for i, (label, risk, fidx) in enumerate(predictions):
            time_sec = fidx / fps
            color = "green" if risk == "Safe" else "red" if "High" in risk else "orange"
            timeline += f"<span style='color:{color};'>■</span> {time_sec:.1f}s: <strong>{label.replace('_',' ').title()}</strong><br>"
            if i >= 20:  # عرض أول 20 فقط
                timeline += f"<i>... and {len(predictions)-20} more</i>"
                break
        st.markdown(f"<div style='max-height:200px; overflow-y:auto; padding:10px; background:#f0f2f6; border-radius:8px;'>{timeline}</div>", unsafe_allow_html=True)

        st.balloons()
    else:
        st.warning("No predictions were made.")

# ==============================================================
# Footer
# ==============================================================
st.markdown("<div class='footer'><p>Smooth Video • Full Analysis • Real-time AI</p></div>", unsafe_allow_html=True)
