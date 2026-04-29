
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL.Image

# Load the trained model
# Make sure 'cnn_classifier.h5' is in the same directory as this app.py file
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cnn_classifier.h5')
    return model

model = load_model()

# Define the class labels based on your training data (assuming the order is consistent)
# You might want to save these labels directly from your training script as well
# For this example, I'm inferring them based on your directory structure setup.
# If you have a different order or more classes, adjust this list.
# Based on your previous output: 'Found 24 images belonging to 3 classes.'
# and the prediction 'Sad', we'll use example classes. 
# It's best practice to save train_data.class_indices during training.
# For now, let's assume the labels are alphabetically ordered or known:
class_labels = ['Angry', 'Happy', 'Sad'] # Adjust this if your actual class order is different

st.title('Dog Emotion Classifier')
st.write('Upload an image of a dog to predict its emotion!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = PIL.Image.open(uploaded_file).convert('RGB') # Ensure image is RGB
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    img = img.resize((128, 128)) # Resize to target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize pixel values

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_emotion = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    st.success(f"Predicted Emotion: **{predicted_emotion}** with **{confidence:.2f}%** confidence")

st.markdown("---")
st.markdown("**Note:** For this app to run locally, ensure you have `streamlit`, `tensorflow`, and `Pillow` installed (`pip install streamlit tensorflow Pillow`). Also, make sure the `cnn_classifier.h5` model file is in the same directory as `app.py`.")
