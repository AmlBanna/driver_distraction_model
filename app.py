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
st.set_page_config(page_title="Driver Behavior AI", page_icon="🚗", layout="wide")

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
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================
# 2. Load model
# ==============================================================
current_dir = os.path.dirname(os.path.abspath(_file_))
model_path = os.path.join(current_dir, "driver_distraction_model.keras")
json_path = os.path.join(current_dir, "class_indices.json")

@st.cache_resource(show_spinner="🔄 Loading AI model...")
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

def process_video_with_labels(input_path, output_path):
    """Process video + add labels + save new video"""
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    predictions = []
    frame_idx = 0
    predict_every = 4  # Predict every 4 frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        
        # Predict every N frames
        label, risk = "safe_driving", "Safe"
        if frame_idx % predict_every == 0:
            cls, conf = predict_once(frame)
            label, risk = get_label(cls, conf)
            predictions.append((label, risk, frame_idx))
        
        # Draw label on frame
        color_map = {
            "safe_driving": (0, 255, 0),      # Green
            "using_phone": (0, 0, 255),       # Red
            "drinking": (200, 0, 200),        # Purple
            "hair_makeup": (255, 20, 147),    # Pink
            "turning": (0, 255, 255),         # Yellow
            "radio": (100, 100, 255),         # Blue
            "other_activity": (150, 150, 150) # Gray
        }
        color = color_map.get(label, (255, 255, 255))
        
        # Big bold label
        cv2.rectangle(frame, (10, 10), (10 + width//2, 90), color, -1)
        cv2.putText(frame, label.replace("_", " ").title(), 
                   (20, 65), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 4)
        cv2.putText(frame, f"Risk: {risk}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        out.write(frame)
    
    cap.release()
    out.release()
    return predictions

# ==============================================================
# 4. Main UI
# ==============================================================
st.markdown("<h1 class='big-title'>🚗 Driver Behavior AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time labels ON VIDEO + Complete analysis below</p>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📁 Upload MP4 video (43 seconds recommended)", type=["mp4"])

if uploaded_file is not None:
    # Progress container
    progress_container = st.container()
    
    # Save original file
    orig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(orig_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process video with labels
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🎬 Processing video + AI analysis...")
        progress_bar.progress(20)
        
        # Create output path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_analyzed.mp4").name
        
        # Process!
        predictions = process_video_with_labels(orig_path, output_path)
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
    
    # === عرض الفيديو المتحرك مع التصنيف ===
    st.markdown("## 🎥 Analyzed Video (Play to see labels)")
    st.video(output_path)
    
    # === الإحصائيات المتقدمة ===
    if predictions:
        col1, col2, col3, col4 = st.columns(4)
        
        labels = [p[0] for p in predictions]
        counter = Counter(labels)
        most_common = counter.most_common(1)[0]
        total_preds = len(predictions)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color:#1E3A8A; margin:0;'>🎯 Most Common</h3>
                <h2 style='color:#1E3A8A; margin:0.5rem 0 0 0;'>{most_common[0].replace('_',' ').title()}</h2>
                <p style='color:#6B7280; margin:0;'>{most_common[1]}x ({most_common[1]/total_preds*100:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            safe_count = counter.get("safe_driving", 0)
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color:#10B981; margin:0;'>✅ Safe Driving</h3>
                <h2 style='color:#10B981; margin:0.5rem 0 0 0;'>{safe_count}x</h2>
                <p style='color:#6B7280; margin:0;'>{safe_count/total_preds*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            danger_count = sum(counter.get(k, 0) for k in ["using_phone", "drinking", "hair_makeup"])
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color:#EF4444; margin:0;'>⚠ High Risk</h3>
                <h2 style='color:#EF4444; margin:0.5rem 0 0 0;'>{danger_count}x</h2>
                <p style='color:#6B7280; margin:0;'>{danger_count/total_preds*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color:#3B82F6; margin:0;'>📊 Total</h3>
                <h2 style='color:#3B82F6; margin:0.5rem 0 0 0;'>{total_preds}</h2>
                <p style='color:#6B7280; margin:0;'>Predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed timeline
        st.markdown("### 📈 Behavior Timeline (First 25)")
        timeline_data = []
        for label, risk, frame_idx in predictions[:25]:
            time_sec = frame_idx / 30  # Assuming ~30fps
            risk_emoji = "🟢" if risk == "Safe" else "🟡" if "Moderate" in risk else "🔴"
            timeline_data.append({
                "Time": f"{time_sec:.1f}s",
                "Behavior": label.replace("_", " ").title(),
                "Risk": risk_emoji
            })
        
        st.dataframe(timeline_data, use_container_width=True)
        
        st.balloons()
    else:
        st.warning("No predictions made.")
    
    # Cleanup
    os.unlink(orig_path)
    os.unlink(output_path)

# ==============================================================
# Footer
# ==============================================================
st.markdown("""
<div class='footer'>
    <p>🚗 Driver Behavior AI | Labels ON VIDEO + Full Analytics | Powered by TensorFlow</p>
</div>
""", unsafe_allow_html=True)
