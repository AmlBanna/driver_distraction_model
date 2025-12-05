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
from gtts import gTTS  # للتنبيه الصوتي
import base64

# ================================
# Page Config & Styling
# ================================
st.set_page_config(page_title="Driver Behavior Detection", page_icon="🚗", layout="centered")
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; font-weight: 700; color: #1E3A8A; text-align: center;}
    .sub-header {font-size: 1.3rem; color: #4B5563; text-align: center; margin-bottom: 2rem;}
    .status-box {padding: 1rem; border-radius: 12px; font-weight: bold; text-align:

    .safe {background-color: #DCFCE7; color: #166534; border: 2px solid #BBF7D0;}
    .danger {background-color: #FECACA; color: #991B1B; border: 2px solid #FCA5A5;}
    .warning {background-color: #FEF3C7; color: #92400E; border: 2px solid #FDE68A;}
    .info {background-color: #DBEAFE; color: #1E40AF; border: 2px solid #BFDBFE;}
    .stButton>button {background-color: #1E3A8A; color: white; border-radius: 8px; padding: 0.6rem 1.2rem;}
    .stButton>button:hover {background-color: #1E40AF;}
    .footer {text-align: center; margin-top: 3rem; color: #6B7280; font-size: 0.9rem;}
    .real-time-status {font-size: 1.5rem; font-weight: bold; text-align: center; padding: 0.8rem; border-radius: 10px; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# ================================
# File Paths
# ================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'driver_distraction_model.keras')
json_path = os.path.join(current_dir, 'class_indices.json')

# ================================
# Load Model
# ================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    if not os.path.exists(model_path):
        st.error("Model file not found! Please upload `driver_distraction_model.keras`")
        st.stop()
    if not os.path.exists(json_path):
        st.error("Class indices file not found! Please upload `class_indices.json`")
        st.stop()

    model = tf.keras.models.load_model(model_path)
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    predict_fn = tf.function(lambda x: model(x, training=False))
    return model, class_indices, idx_to_class, predict_fn

model, class_indices, idx_to_class, predict_fn = load_model()

# ================================
# Classification Logic
# ================================
def get_final_label(cls, conf):
    if cls == 'c6' and conf > 0.30:
        return 'drinking', 'High Risk'
    if cls in ['c1', 'c2', 'c3', 'c4', 'c9'] and conf > 0.28:
        return 'using_phone', 'High Risk'
    if cls == 'c0' and conf > 0.5:
        return 'safe_driving', 'Safe'
    if cls == 'c7' and conf > 0.7:
        return 'turning', 'Moderate Risk'
    if cls == 'c8' and conf > 0.8:
        return 'hair_makeup', 'High Risk'
    if cls == 'c5' and conf > 0.6:
        return 'radio', 'Moderate Risk'
    return 'other_activity', 'Unknown'

# ================================
# Preprocessing
# ================================
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ================================
# Prediction with History & Alert
# ================================
history = []
frame_count = 0
skip_frames = 1
alert_triggered = False
alert_start_time = None

def predict_smooth(frame):
    global history, frame_count, alert_triggered, alert_start_time

    frame_count += 1
    if frame_count % (skip_frames + 1) != 0:
        if history:
            return Counter([h[0] for h in history]).most_common(1)[0][0]
        return 'safe_driving', 'Safe'

    input_tensor = tf.convert_to_tensor(preprocess(frame))
    pred = predict_fn(input_tensor)[0].numpy()
    idx = np.argmax(pred)
    cls = idx_to_class[idx]
    conf = pred[idx]

    label, risk = get_final_label(cls, conf)
    history.append((label, risk, time.time()))
    if len(history) > 10:
        history.pop(0)

    # Smoothing
    if len(history) >= 3:
        most_common = Counter([h[0] for h in history]).most_common(1)[0][0]
        risk_common = Counter([h[1] for h in history if h[0] == most_common]).most_common(1)[0][0]
        return most_common, risk_common
    return label, risk

# ================================
# Alert System (3 sec non-safe)
# ================================
def check_alert(history):
    if len(history) < 3:
        return False, None
    recent = [h[0] for h in history[-3:]]
    if all(r != 'safe_driving' for r in recent):
        return True, recent[-1]
    return False, None

# ================================
# Play Alert Sound
# ================================
def play_alert_sound(label):
    text = f"Warning! Driver is {label.replace('_', ' ')}!"
    tts = gTTS(text, lang='en')
    tts.save("alert.mp3")
    audio_file = open("alert.mp3", 'rb')
    audio_bytes = audio_file.read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay="true" style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ================================
# UI
# ================================
st.markdown("<h1 class='main-header'>Driver Behavior Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Real-time AI | Alerts | Stats | Image Upload</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.info("Model: CNN (224x224)\nClasses: 10 behaviors\nReal-time alerts")
    st.markdown("### Risk Levels")
    st.markdown("- Safe")
    st.markdown("- Moderate Risk")
    st.markdown("- High Risk")

tab1, tab2, tab3 = st.tabs(["Live Camera", "Upload Video", "Upload Image"])

# ================================
# Tab 1: Live Camera
# ================================
with tab1:
    st.write("#### Live Real-Time Detection")
    start_btn = st.button("Start Camera", type="primary", key="cam_start")
    stop_btn = st.button("Stop", type="secondary", key="cam_stop")

    if 'cam_running' not in st.session_state:
        st.session_state.cam_running = False

    if start_btn:
        st.session_state.cam_running = True
        history.clear()
        alert_triggered = False

    if st.session_state.cam_running:
        stframe = st.empty()
        status_box = st.empty()
        alert_box = st.empty()
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and st.session_state.cam_running:
            ret, frame = cap.read()
            if not ret: break

            label, risk = predict_smooth(frame)

            # Real-time status
            status_text = f"Currently: {label.replace('_', ' ').title()}"
            status_color = "safe" if risk == "Safe" else "warning" if "Moderate" in risk else "danger"
            status_box.markdown(f"<div class='real-time-status {status_color}'>{status_text}</div>", unsafe_allow_html=True)

            # Draw on frame
            color = (0, 255, 0) if label == 'safe_driving' else (0, 0, 255)
            if 'drinking' in label: color = (200, 0, 200)
            if 'hair_makeup' in label: color = (255, 20, 147)
            cv2.putText(frame, label.replace('_', ' ').title(), (15, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 4)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Alert check
            alert, alert_label = check_alert(history)
            if alert and not alert_triggered:
                alert_triggered = True
                alert_box.error(f"ALERT: Driver has been {alert_label.replace('_', ' ')} for 3 seconds!")
                play_alert_sound(alert_label)
            elif not alert:
                alert_triggered = False

        cap.release()
        st.session_state.cam_running = False
        st.success("Camera stopped.")

# ================================
# Tab 2: Upload Video
# ================================
with tab2:
    st.write("#### Upload Video for Analysis")
    uploaded_file = st.file_uploader("Choose video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], key="video")

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name

        cap = cv2.VideoCapture(tfile_path)
        stframe = st.empty()
        status_box = st.empty()
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0
        history.clear()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            label, risk = predict_smooth(frame)
            status_box.markdown(f"<div class='real-time-status {'safe' if risk=='Safe' else 'danger'}'>Detected: {label.replace('_', ' ').title()}</div>", unsafe_allow_html=True)

            color = (0, 255, 0) if label == 'safe_driving' else (0, 0, 255)
            if 'drinking' in label: color = (200, 0, 200)
            cv2.putText(frame, label.replace('_', ' ').title(), (15, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 4)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            processed += 1
            progress_bar.progress(processed / total_frames)

        cap.release()
        os.unlink(tfile_path)

        # Final Stats
        if history:
            behaviors = [h[0] for h in history]
            most_common = Counter(behaviors).most_common(1)[0]
            st.markdown(f"### Final Report")
            st.markdown(f"**Most frequent behavior**: `{most_common[0].replace('_', ' ').title()}` ({most_common[1]} times)")
            st.balloons()
        else:
            st.info("No frames processed.")

# ================================
# Tab 3: Upload Image
# ================================
with tab3:
    st.write("#### Upload Image for Instant Analysis")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img")

    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            label, risk = predict_smooth(img)
            color = (0, 255, 0) if label == 'safe_driving' else (0, 0, 255)
            if 'drinking' in label: color = (200, 0, 200)
            cv2.putText(img, label.replace('_', ' ').title(), (15, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 4)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Result: {label.replace('_', ' ').title()} | Risk: {risk}")
            st.markdown(f"<div class='status-box {'safe' if risk=='Safe' else 'danger'}'>Result: <strong>{label.replace('_', ' ').title()}</strong> | Risk: <strong>{risk}</strong></div>", unsafe_allow_html=True)
        else:
            st.error("Invalid image.")

# ================================
# Footer
# ================================
st.markdown("""
<div class='footer'>
    <p>Driver Behavior Detection System | Real-time Alerts | Final Stats | Image Support</p>
</div>
""", unsafe_allow_html=True)
