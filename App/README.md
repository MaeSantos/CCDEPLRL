# Dog Emotion App

Native Flutter Android app for live dog emotion classification.

## Model format

Load a `.tflite` model from the in-app picker. If your model is currently `.h5`, convert it to TensorFlow Lite first.

The model must:

- Accept image input shaped `[1, height, width, 3]`
- Use RGB images normalized to `[-1, 1]`
- Output 3 scores in this order: `Angry`, `Happy`, `Sad`

## Run

Open this `App` folder in Android Studio, let Flutter/Gradle sync, then run the Android configuration.
