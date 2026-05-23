# Canine Mood - Dog Emotion Scanner

**Canine Mood** is a modern, AI-powered Android application designed to detect and interpret your dog's emotions in real-time. Using a custom TensorFlow Lite classification model, the app analyzes live camera feeds to identify if a dog is feeling **Happy**, **Sad**, or **Angry**.

<p align="center">
  <img src="logo/Logo.png" width="200" alt="Canine Mood Logo">
</p>

## 🚀 Key Features

*   **Real-time Emotion Detection**: Instantly classifies dog emotions into three categories: Happy, Sad, and Angry.
*   **Dual Camera Support**: Easily switch between **Front** and **Back** cameras to scan your dog from any angle.
*   **Interactive Sound Effects**: The app reacts to your dog's mood with high-quality sound effects:
    *   😊 **Happy**: Plays a joyful happy bark.
    *   😢 **Sad**: Plays a mournful sad howl.
    *   😠 **Angry**: Plays a protective angry bark.
*   **Voice Feedback**: Optionally speaks the detected emotion using Android's Text-to-Speech engine.
*   **Modern UI**: Features a clean, card-based design with smooth transitions and status indicators.
*   **Privacy-First**: All processing is done locally on the device. No camera data is sent to the cloud, and no internet connection is required.

## 🛠️ Technical Details

*   **Engine**: Native Android (Java)
*   **Camera API**: Android Camera2 API for high-performance frame capture.
*   **AI Framework**: TensorFlow Lite for efficient on-device inference.
*   **Audio Engine**: SoundPool for low-latency, responsive audio playback.
*   **Model**: Custom MobileNetV2-based TFLite classifier.

## 📂 Project Structure

*   `app/src/main/assets/model.tflite`: The bundled AI model.
*   `app/src/main/res/raw/`: Audio assets for Happy, Sad, and Angry reactions.
*   `app/src/main/java/com/dogemotion/scanner/MainActivity.java`: Core application logic and UI orchestration.

## 📦 How to Build

1.  Open the `android-app` folder in **Android Studio**.
2.  Allow Gradle to synchronize dependencies.
3.  Ensure you have a physical Android device or an emulator with camera support connected.
4.  Click **Run** or use the terminal:
    ```bash
    ./gradlew assembleDebug
    ```
5.  The generated APK will be located at:
    `app/build/outputs/apk/debug/app-debug.apk`

## 📝 Usage Notes

*   The app requires **Camera Permission** to function.
*   For best results, aim the camera directly at the dog's face in well-lit conditions.
*   The **Switch Camera** button allows you to use the front camera for "dog selfies" or the back camera for standard scanning.

---
*Developed as part of the Dog Emotion Scanner Project.*
