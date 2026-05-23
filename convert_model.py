import tensorflow as tf
import os

print("Loading model...")
model = tf.keras.models.load_model('mobilenet_dog_emotion_classifier.h5')
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

output_path = 'App/assets/model.tflite'
print(f"Saving to {output_path}...")
with open(output_path, 'wb') as f:
    f.write(tflite_model)
print("Done!")
