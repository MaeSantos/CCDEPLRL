"""
Dog Emotion Classifier — Streamlit App
======================================
Real-time webcam + image upload support.

Deploy:
  streamlit run app.py

Files needed in the same directory:
  ├── app.py
  ├── dog_emotion_model.h5
  ├── class_indices.json
  └── requirements.txt
"""

import json
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐶 Dog Emotion Detector",
    page_icon="🐶",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH         = Path("dog_emotion_model.h5")
CLASS_INDICES_PATH = Path("class_indices.json")
IMG_SIZE           = (224, 224)

EMOTION_COLORS = {
    "Happy": (72, 199, 116),    # green
    "Sad":   (66, 133, 244),    # blue
    "Angry": (234, 67, 53),     # red
}
EMOTION_EMOJIS = {
    "Happy": "😄",
    "Sad":   "😢",
    "Angry": "😠",
}

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_classes():
    if not MODEL_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    # invert: index → class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, IDX_TO_CLASS = load_model_and_classes()
CLASS_NAMES = list(IDX_TO_CLASS.values()) if IDX_TO_CLASS else []

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(img_rgb: np.ndarray):
    """img_rgb: uint8 H×W×3 numpy array. Returns (label, confidence, probs_dict)."""
    img   = cv2.resize(img_rgb, IMG_SIZE)
    arr   = img.astype("float32") / 255.0
    arr   = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = IDX_TO_CLASS.get(idx, str(idx))
    conf  = float(probs[idx])
    probs_dict = {IDX_TO_CLASS.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    return label, conf, probs_dict


def overlay_prediction(frame_bgr: np.ndarray, label: str, conf: float) -> np.ndarray:
    """Draw a nice overlay on the OpenCV BGR frame."""
    h, w = frame_bgr.shape[:2]
    color = EMOTION_COLORS.get(label, (255, 255, 255))

    # semi-transparent top bar
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

    text  = f"{EMOTION_EMOJIS.get(label, '')} {label}  {conf*100:.0f}%"
    cv2.putText(
        frame_bgr, text, (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
        color, 2, cv2.LINE_AA
    )

    # colored border
    cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), color, 4)
    return frame_bgr

# ── WebRTC video processor ────────────────────────────────────────────────────
class DogEmotionProcessor:
    def __init__(self):
        self._result = ("—", 0.0, {})
        self._last_run = 0.0
        self._interval = 0.25        # predict every 250 ms

    @property
    def result(self):
        return self._result

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        now = time.time()
        if model is not None and (now - self._last_run) >= self._interval:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            label, conf, probs = predict(img_rgb)
            self._result = (label, conf, probs)
            self._last_run = now

        label, conf, _ = self._result
        if label != "—":
            img_bgr = overlay_prediction(img_bgr, label, conf)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐶 Dog Emotion Detector")
st.markdown(
    "Detects **Happy 😄 · Sad 😢 · Angry 😠** from dog images or live webcam."
)

if model is None:
    st.error(
        "⚠️  Model file not found.  \n"
        "Place `dog_emotion_model.h5` and `class_indices.json` in the same "
        "directory as `app.py` and restart."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_webcam, tab_upload = st.tabs(["📷 Live Webcam", "🖼️ Upload Image"])

# ─────────────────────────── TAB 1: Webcam ────────────────────────────────────
with tab_webcam:
    st.markdown("Allow camera access, then point it at a dog!")

    processor = DogEmotionProcessor()

    ctx = webrtc_streamer(
        key="dog-emotion",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    if ctx.state.playing:
        label_placeholder = st.empty()
        bar_placeholder   = st.empty()

        # live result readout (refreshes while streaming)
        while ctx.state.playing:
            label, conf, probs_dict = processor.result
            with label_placeholder.container():
                if label != "—":
                    emoji = EMOTION_EMOJIS.get(label, "")
                    st.markdown(
                        f"### {emoji} **{label}** &nbsp; `{conf*100:.1f}%`",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("### Waiting for prediction…")

            with bar_placeholder.container():
                if probs_dict:
                    for cls, prob in probs_dict.items():
                        st.progress(
                            float(prob),
                            text=f"{EMOTION_EMOJIS.get(cls,'')} {cls}: {prob*100:.1f}%",
                        )
            time.sleep(0.3)

# ─────────────────────────── TAB 2: Upload ────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a dog photo", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

        with col2:
            with st.spinner("Predicting…"):
                label, conf, probs_dict = predict(img_rgb)

            emoji = EMOTION_EMOJIS.get(label, "")
            st.markdown(f"## {emoji} {label}")
            st.markdown(f"**Confidence:** `{conf*100:.1f}%`")
            st.divider()
            st.markdown("**All probabilities:**")
            for cls, prob in sorted(probs_dict.items(), key=lambda x: -x[1]):
                st.progress(
                    float(prob),
                    text=f"{EMOTION_EMOJIS.get(cls,'')} {cls}: {prob*100:.1f}%",
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: MobileNetV2 fine-tuned on Dog Emotions dataset (Happy / Sad / Angry). "
    "Built with TensorFlow + Streamlit."
)
