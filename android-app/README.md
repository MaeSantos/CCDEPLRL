# Dog Emotion Scanner Android App

This Android project wraps the existing HTML scanner in a native WebView shell.

## Build

Open this `android-app` folder in Android Studio, let Gradle sync, then run the `app` configuration.

The debug APK built locally here:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## Notes

- The app requests camera permission for live scanning.
- The app requests internet permission because the HTML loads TensorFlow.js and fonts from CDNs.
- Model file selection is handled through Android's native file picker.
