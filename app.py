# ============================================================
#  Dog Emotion Classifier — Streamlit App (Modern UI Edition)
#  app.py
#
#  Setup:
#    1. pip install streamlit tensorflow pillow
#    2. Place dog_emotion_model.h5 and class_labels.json
#       in the same folder as this file
#    3. streamlit run app.py
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import time
from PIL import Image

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dog Emotion Classifier",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for a Modern Look ──────────────────────────────
# Hides default Streamlit branding and adds modern shadow cards
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern card styling for the prediction result */
    .modern-card {
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
        margin-bottom: 24px;
        text-align: center;
    }
    .modern-card:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & labels (cached) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model  = tf.keras.models.load_model("dog_emotion_model.h5")
    with open("class_labels.json") as f:
        labels = json.load(f)
    return model, labels

# ── Preprocess image ─────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 128, 128, 3)

# ── Emoji & color per class ──────────────────────────────────
EMOJI  = {"Angry": "😠", "Happy": "😄", "Sad": "😢"}
COLOR  = {"Angry": "#FF4B4B", "Happy": "#21C354", "Sad": "#1C83E1"}

# ── Sidebar (Declutters main UI) ─────────────────────────────
with st.sidebar:
    st.title("🐶 App Info")
    st.markdown("""
    Welcome to the **Dog Emotion Classifier**. 
    
    This app uses a Convolutional Neural Network (CNN) to analyze canine facial expressions and predict how your dog is feeling.
    """)
    st.divider()
    st.caption("Developed for: Deep Learning Project")

# ── Main UI Header ───────────────────────────────────────────
st.title("Dog Emotion Classifier")
st.markdown("Discover what your furry friend is feeling. Choose an input method below to begin.")

# Load model with subtle error handling
try:
    with st.spinner("Waking up the AI..."):
        model, CLASS_LABELS = load_model()
except Exception as e:
    st.error("⚠️ Model files missing. Please ensure `dog_emotion_model.h5` and `class_labels.json` are present.")
    st.stop()

# ── Input Section (Tabs) ─────────────────────────────────────
st.write("") # Spacer
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Live Webcam"])

image_source = None

with tab1:
    uploaded = st.file_uploader(
        "Drag and drop a dog photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        image_source = uploaded

with tab2:
    camera = st.camera_input("Take a picture of a dog", label_visibility="collapsed")
    if camera:
        image_source = camera

# ── Prediction & Results Section ─────────────────────────────
if image_source:
    img = Image.open(image_source)

    st.divider()
    st.markdown("### Analysis Results")
    
    # Modern two-column layout
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.image(img, caption="Subject Image", use_container_width=True, output_format="JPEG")

    with col2:
        with st.spinner("Analyzing canine facial features..."):
            # Artificial slight delay so the user registers that work is happening (feels more like a real app)
            time.sleep(0.5) 
            arr   = preprocess(img)
            preds = model.predict(arr, verbose=0)[0]

        top_idx   = int(np.argmax(preds))
        top_label = CLASS_LABELS[top_idx]
        top_conf  = float(preds[top_idx]) * 100

        emoji = EMOJI.get(top_label, "🐾")
        color = COLOR.get(top_label, "#888")

        # Top Prediction Card (Modernized with CSS classes and softer colors)
        st.markdown(f"""
        <div class="modern-card" style="background-color: {color}15; border: 1px solid {color}40;">
            <div style="font-size: 56px; line-height: 1.2;">{emoji}</div>
            <div style="font-size: 28px; font-weight: 700; color: {color}; letter-spacing: -0.5px;">
                {top_label}
            </div>
            <div style="font-size: 16px; color: #555; margin-top: 8px; font-weight: 500;">
                {top_conf:.1f}% Match
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Success toast notification for a dynamic feel
        st.toast(f"Analysis complete! Looks like a {top_label} dog.", icon=emoji)

        # Confidence Progress Bars
        st.markdown("<h4 style='font-size: 16px; margin-bottom: 0px;'>Confidence Breakdown</h4>", unsafe_allow_html=True)
        for label, prob in zip(CLASS_LABELS, preds):
            pct = float(prob) * 100
            label_emoji = EMOJI.get(label, "🐾")
            # Using st.progress combined with columns for a cleaner text layout
            p_col1, p_col2 = st.columns([1, 3])
            with p_col1:
                st.markdown(f"**{label_emoji} {label}**")
            with p_col2:
                st.progress(float(prob))

    # Raw probabilities table (collapsible, kept out of the way)
    st.write("")
    with st.expander("View Raw Model Output"):
        for label, prob in zip(CLASS_LABELS, preds):
            st.write(f"**{label}**: `{prob:.6f}`")

else:
    # Modern empty state
    st.info("👆 Provide an image using the tabs above to see the AI in action.")