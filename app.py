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
# 1. Page config + Fancy CSS
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
    .info {background:#DBEAFE; color:#1E40AF; border:2px solid #BFDBFE;}
    .real-time {font-size:1.6rem; font-weight:bold; text-align:center; padding:0.6rem; border-radius:10px; margin:0.8rem 0;}
    .stButton>button {background:#1E3A8A; color:white; border-radius:8px; padding:0.6rem 1.2rem;}
    .stButton>button:hover {background:#1E40AF;}
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
    if not os.path.exists(model_path):
        st.error("`driver_distraction_model.keras` not found!")
        st.stop()
    if not os.path.exists(json_path):
        st.error("`class_indices.json` not found!")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    with open(json_path) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    predict_fn = tf.function(lambda x: model(x, training=False))
    return model, class_indices, idx_to_class, predict_fn

model, class_indices, idx_to_class, predict_fn = load_model()

# ==============================================================
# 3. Classification + Preprocess
# ==============================================================
def get_final_label(cls, conf):
    if cls == "c6" and conf > 0.30: return "drinking", "High Risk"
    if cls in ["c1", "c2", "c3", "c4", "c9"] and conf > 0.28: return "using_phone", "High Risk"
    if cls == "c0" and conf > 0.5: return "safe_driving", "Safe"
    if cls == "c7" and conf > 0.7: return "turning", "Moderate Risk"
    if cls == "c8" and conf > 0.8: return "hair_makeup", "High Risk"
    if cls == "c5" and conf > 0.6: return "radio", "Moderate Risk"
    return "other_activity", "Unknown"

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ==============================================================
# 4. Prediction (always return 2 values)
# ==============================================================
history = []
frame_counter = 0
skip_frames = 1

def predict_smooth(frame):
    global history, frame_counter
    frame_counter += 1

    if frame_counter % (skip_frames + 1) != 0:
        if history:
            most = Counter([h[0] for h in history]).most_common(1)[0][0]
            risk = next((h[1] for h in history if h[0] == most), "Safe")
            return most, risk
        return "safe_driving", "Safe"

    inp = tf.convert_to_tensor(preprocess(frame))
    pred = predict_fn(inp)[0].numpy()
    idx = np.argmax(pred)
    cls = idx_to_class[idx]
    conf = pred[idx]
    label, risk = get_final_label(cls, conf)

    history.append((label, risk, time.time()))
    if len(history) > 12: history.pop(0)

    if len(history) >= 3:
        most = Counter([h[0] for h in history]).most_common(1)[0][0]
        risk = Counter([h[1] for h in history if h[0] == most]).most_common(1)[0][0]
        return most, risk
    return label, risk

# ==============================================================
# 5. Alert
# ==============================================================
def check_alert():
    if len(history) < 3: return False, None
    recent = [h[0] for h in history[-3:]]
    if all(r != "safe_driving" for r in recent):
        return True, recent[-1]
    return False, None

def play_alert(label):
    txt = f"Warning! Driver is {label.replace('_', ' ')}!"
    tts = gTTS(txt, lang="en")
    tts.save("alert.mp3")
    with open("alert.mp3", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# ==============================================================
# 6. UI
# ==============================================================
st.markdown("<h1 class='big-title'>Driver Behavior AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time • Alerts • Stats • Image upload</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.info("CNN 224×224 | 10 classes")
    st.markdown("### Risk Levels")
    st.markdown("- Safe | - Moderate Risk | - High Risk")

tab_cam, tab_vid, tab_img = st.tabs(["Live Camera", "Upload Video", "Upload Image"])

# --- Live Camera ---
with tab_cam:
    st.write("#### Real-time detection")
    col1, col2 = st.columns([3, 1])
    with col2:
        start = st.button("Start", type="primary", key="cam_start")
        stop = st.button("Stop", type="secondary", key="cam_stop")

    if start: st.session_state.cam = True
    if stop: st.session_state.cam = False

    if getattr(st.session_state, "cam", False):
        frame_ph = st.empty()
        status_ph = st.empty()
        alert_ph = st.empty()
        cap = cv2.VideoCapture(0)
        alert_shown = False

        while cap.isOpened() and st.session_state.cam:
            ret, frame = cap.read()
            if not ret: break

            label, risk = predict_smooth(frame)
            status_color = "safe" if risk == "Safe" else "warning" if "Moderate" in risk else "danger"
            status_ph.markdown(f"<div class='real-time {status_color}'>Current: {label.replace('_', ' ').title()}</div>", unsafe_allow_html=True)

            col_map = {"safe_driving": (0,255,0), "using_phone": (0,0,255), "drinking": (200,0,200), "hair_makeup": (255,20,147), "turning": (0,255,255), "radio": (100,100,255), "other_activity": (150,150,150)}
            cv2.putText(frame, label.replace("_", " ").title(), (15,70), cv2.FONT_HERSHEY_DUPLEX, 2.0, col_map.get(label, (255,255,255)), 4)
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            alert, alabel = check_alert()
            if alert and not alert_shown:  # تم التصحيح
                alert_shown = True
                alert_ph.error(f"ALERT: {alabel.replace('_', ' ')} for 3 seconds!")
                play_alert(alabel)
            elif not alert:
                alert_shown = False

        cap.release()
        st.session_state.cam = False
        st.success("Camera stopped.")

# --- Upload Video ---
with tab_vid:
    st.write("#### Analyse video")
    vid_file = st.file_uploader("MP4 / AVI / MOV", type=["mp4", "avi", "mov"], key="vid")
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid_file.name)[1])
        tfile.write(vid_file.read())
        tpath = tfile.name
        cap = cv2.VideoCapture(tpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prog = st.progress(0)
        frame_ph = st.empty()
        status_ph = st.empty()
        history.clear()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            label, risk = predict_smooth(frame)
            status_ph.markdown(f"<div class='real-time {'safe' if risk=='Safe' else 'danger'}'>Detected: {label.replace('_',' ').title()}</div>", unsafe_allow_html=True)
            col = (0,255,0) if label == "safe_driving" else (0,0,255)
            if "drinking" in label: col = (200,0,200)
            cv2.putText(frame, label.replace("_", " ").title(), (15,70), cv2.FONT_HERSHEY_DUPLEX, 2.0, col, 4)
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            prog.progress(cap.get(cv2.CAP_PROP_POS_FRAMES) / total)

        cap.release()
        os.unlink(tpath)

        if history:
            beh = [h[0] for h in history]
            most = Counter(beh).most_common(1)[0]
            st.markdown(f"### Final Report")
            st.markdown(f"<div class='stats-box'><strong>Most frequent:</strong> {most[0].replace('_',' ').title()} ({most[1]} frames)</div>", unsafe_allow_html=True)
            st.balloons()

# --- Upload Image ---
with tab_img:
    st.write("#### Instant image analysis")
    img_file = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"], key="img")
    if img_file:
        nparr = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Invalid image")
        else:
            label, risk = predict_smooth(img)
            col = (0,255,0) if label == "safe_driving" else (0,0,255)
            if "drinking" in label: col = (200,0,200)
            cv2.putText(img, label.replace("_", " ").title(), (15,70), cv2.FONT_HERSHEY_DUPLEX, 2.0, col, 4)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{label.replace('_',' ').title()} – {risk}")
            st.markdown(f"<div class='status-card {'safe' if risk=='Safe' else 'danger'}'>Result: {label.replace('_',' ').title()} | Risk: {risk}</div>", unsafe_allow_html=True)

# ==============================================================
# Footer
# ==============================================================
st.markdown("<div class='footer'><p>Driver Behavior AI • Real-time • Alerts • Stats</p></div>", unsafe_allow_html=True)
