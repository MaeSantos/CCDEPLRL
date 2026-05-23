# CCDEPLRL - Dog Emotion Classification

This repository contains a dog emotion classification project with training notebooks, a Python app, a web-based classifier, and Android app versions for testing the model on a device.

## Project Overview

The goal of this project is to classify dog emotions from images or camera input. The main emotion labels used by the app are:

- Angry
- Happy
- Sad

The repository includes model experimentation files, app source code, and mobile implementations that can be used to demonstrate the classifier.

## Repository Contents

- `Before_After_Optimization.ipynb` - notebook showing model optimization work
- `Prodi_Santos_CNN.ipynb` - CNN model development notebook
- `app.py` - Python application entry point
- `class_labels.json` - emotion class labels used by the model
- `requirements.txt` - Python dependencies
- `DogEmotionClassifier.html` - browser-based dog emotion classifier
- `mobilenet_dog_emotion_classifier.h5` - trained Keras model file
- `App/` - native Flutter Android app
- `android-app/` - Android WebView wrapper for the HTML classifier

## Flutter Android App

The `App` folder contains a native Flutter Android app for live dog emotion classification.

To run it:

1. Open the `App` folder in Android Studio.
2. Let Flutter and Gradle sync the project.
3. Run the Android configuration on an emulator or physical Android device.

The Flutter app expects a TensorFlow Lite model selected from the in-app file picker. If the model is still in `.h5` format, convert it to `.tflite` first.

Expected model format:

- Input shape: `[1, height, width, 3]`
- RGB image input
- Normalized image values in the range `[-1, 1]`
- Output order: `Angry`, `Happy`, `Sad`

## Android WebView App

The `android-app` folder wraps the HTML classifier in a native Android WebView shell.

To run it:

1. Open the `android-app` folder in Android Studio.
2. Let Gradle sync the project.
3. Run the `app` configuration.

The Android WebView app requests camera permission for live scanning and internet permission because the HTML page loads TensorFlow.js and other web resources from CDNs.

## Model File

The trained model file is included as:

```text
mobilenet_dog_emotion_classifier.h5
```

Original shared model/resource link:

```text
https://drive.google.com/file/d/15lkXMflPdLr-6zURW2Ri6sFt1gvqJcTc/view?usp=sharing
```

## Authors

Mae Santos
