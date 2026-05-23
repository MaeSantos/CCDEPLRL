package com.dogemotion.scanner;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import android.graphics.drawable.GradientDrawable;
import android.media.AudioAttributes;
import android.media.SoundPool;
import android.util.Log;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int CAMERA_REQUEST_CODE = 1001;
    private static final String[] LABELS = {"Angry", "Happy", "Sad"};
    private static final int[] COLORS = {0xffff7043, 0xffffd54f, 0xff4db6ac};
    private static final long INFERENCE_INTERVAL_MS = 250;
    private static final float SPEECH_CONFIDENCE = 0.65f;

    private TextureView previewView;
    private TextView statusText;
    private TextView emotionText;
    private TextView confidenceText;
    private TextView modelText;
    private TextView statsText;
    private TextView logText;
    private ProgressBar[] scoreBars;
    private TextView[] scoreLabels;
    private Button cameraButton;
    private Button switchCameraButton;

    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private ImageReader imageReader;
    private String cameraId;

    private Interpreter interpreter;
    private TextToSpeech tts;
    private SoundPool soundPool;
    private int[] soundIds = new int[3]; // Happy, Sad, Angry
    private int inputWidth = 224;
    private int inputHeight = 224;
    private int lensFacing = CameraCharacteristics.LENS_FACING_BACK;
    private boolean modelReady = false;
    private boolean cameraRunning = false;
    private boolean processingFrame = false;
    private boolean voiceEnabled = true;
    private long lastInferenceAt = 0;
    private long lastSpeechAt = 0;
    private long lastSoundAt = 0;
    private String lastSpokenEmotion = "";
    private String lastPlayedEmotion = "";
    private int frameCount = 0;
    private float lastInferenceMs = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        initSoundPool();
        buildUi();
        startBackgroundThread();
        configureTextToSpeech();
        loadModel();

        if (hasCameraPermission()) {
            status("Camera ready");
        } else {
            status("Camera permission needed");
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
        }
    }

    private void buildUi() {
        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        scrollView.setBackgroundColor(0xfffafafa);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(20), dp(20), dp(24));
        scrollView.addView(root, new ScrollView.LayoutParams(
                ScrollView.LayoutParams.MATCH_PARENT,
                ScrollView.LayoutParams.WRAP_CONTENT
        ));

        // Header with Logo and Title
        LinearLayout header = new LinearLayout(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setOrientation(LinearLayout.VERTICAL);
        root.addView(header, matchWrap());

        ImageView logoView = new ImageView(this);
        int logoResId = getResources().getIdentifier("logo_main", "drawable", getPackageName());
        if (logoResId != 0) {
            logoView.setImageResource(logoResId);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(120), dp(120));
            lp.gravity = Gravity.CENTER_HORIZONTAL;
            header.addView(logoView, lp);
        }

        TextView title = new TextView(this);
        title.setText("CANINE MOOD");
        title.setTextColor(0xff212121);
        title.setTextSize(28);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        title.setTypeface(null, 1);
        header.addView(title, withTopMargin(dp(8)));

        TextView subtitle = new TextView(this);
        subtitle.setText("EMOTION DETECTION APP");
        subtitle.setTextColor(0xff757575);
        subtitle.setTextSize(14);
        subtitle.setLetterSpacing(0.1f);
        subtitle.setGravity(Gravity.CENTER_HORIZONTAL);
        header.addView(subtitle, withTopMargin(dp(2)));

        statusText = pill("Loading");
        LinearLayout.LayoutParams pillLp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        pillLp.gravity = Gravity.CENTER_HORIZONTAL;
        pillLp.topMargin = dp(12);
        header.addView(statusText, pillLp);

        // Preview Area
        LinearLayout previewContainer = panel(root);
        previewContainer.setPadding(dp(4), dp(4), dp(4), dp(4));
        
        previewView = new TextureView(this);
        previewView.setSurfaceTextureListener(surfaceListener);
        previewContainer.addView(previewView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(320)
        ));

        cameraButton = styledButton("START SCANNING", 0xff212121);
        cameraButton.setOnClickListener(view -> toggleCamera());
        root.addView(cameraButton, withTopMargin(dp(16)));

        switchCameraButton = styledButton("SWITCH CAMERA", 0xff757575);
        switchCameraButton.setOnClickListener(view -> switchCamera());
        root.addView(switchCameraButton, withTopMargin(dp(10)));

        // Result Card
        emotionText = cardText(root, "AWAITING DOG", 32, 0xffbdbdbd, Gravity.CENTER);
        confidenceText = cardText(root, "Position camera to see your dog's mood", 14, 0xff757575, Gravity.CENTER);

        // Score Panel
        scoreBars = new ProgressBar[LABELS.length];
        scoreLabels = new TextView[LABELS.length];
        LinearLayout scorePanel = panel(root);
        TextView scoresTitle = new TextView(this);
        scoresTitle.setText("DETECTION PROBABILITY");
        scoresTitle.setTextSize(12);
        scoresTitle.setTextColor(0xff9e9e9e);
        scoresTitle.setTypeface(null, 1);
        scorePanel.addView(scoresTitle, withTopMargin(0));

        for (int i = 0; i < LABELS.length; i++) {
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.VERTICAL);
            scorePanel.addView(row, withTopMargin(dp(12)));

            scoreLabels[i] = new TextView(this);
            scoreLabels[i].setText(LABELS[i].toUpperCase(Locale.US) + " - 0%");
            scoreLabels[i].setTextColor(0xff424242);
            scoreLabels[i].setTextSize(13);
            scoreLabels[i].setTypeface(null, 1);
            row.addView(scoreLabels[i]);

            ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
            bar.setMax(1000);
            bar.setProgress(0);
            bar.setProgressTintList(android.content.res.ColorStateList.valueOf(COLORS[i]));
            scoreBars[i] = bar;
            row.addView(bar, matchWrap());
        }

        // Voice Control
        LinearLayout voicePanel = panel(root);
        voicePanel.setOrientation(LinearLayout.HORIZONTAL);
        voicePanel.setGravity(Gravity.CENTER_VERTICAL);
        
        TextView voiceLabel = new TextView(this);
        voiceLabel.setText("VOICE FEEDBACK");
        voiceLabel.setTextColor(0xff424242);
        voiceLabel.setTypeface(null, 1);
        voicePanel.addView(voiceLabel, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));

        Button voiceButton = styledButton("ON", 0xff4caf50);
        voiceButton.setPadding(dp(16), dp(8), dp(16), dp(8));
        voiceButton.setOnClickListener(view -> {
            voiceEnabled = !voiceEnabled;
            voiceButton.setText(voiceEnabled ? "ON" : "OFF");
            voiceButton.setBackground(roundedRect(voiceEnabled ? 0xff4caf50 : 0xff757575, 4));
            if (!voiceEnabled && tts != null) {
                tts.stop();
            }
        });
        voicePanel.addView(voiceButton);

        modelText = new TextView(this);
        modelText.setText("Model: loading");
        modelText.setTextSize(11);
        modelText.setTextColor(0xffbdbdbd);
        modelText.setGravity(Gravity.CENTER);
        root.addView(modelText, withTopMargin(dp(20)));

        statsText = new TextView(this);
        statsText.setText("Ready");
        statsText.setTextSize(11);
        statsText.setTextColor(0xffbdbdbd);
        statsText.setGravity(Gravity.CENTER);
        root.addView(statsText, withTopMargin(dp(4)));

        logText = new TextView(this);
        logText.setVisibility(View.GONE); // Hide system log from main UI
        
        setContentView(scrollView);
    }

    private void configureTextToSpeech() {
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tts.setLanguage(Locale.US);
                tts.setSpeechRate(0.45f);
            }
        });
    }

    private void loadModel() {
        try {
            interpreter = new Interpreter(loadMappedAsset("model.tflite"));
            int[] inputShape = interpreter.getInputTensor(0).shape();
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            if (inputShape.length != 4 || inputShape[3] != 3 || outputShape[outputShape.length - 1] != LABELS.length) {
                throw new IllegalStateException("Unsupported model shape");
            }
            inputHeight = inputShape[1];
            inputWidth = inputShape[2];
            modelReady = true;
            modelText.setText("Model: built-in TFLite classifier");
            confidenceText.setText("Start camera, aim at dog");
            log("Model ready: " + inputWidth + "x" + inputHeight);
        } catch (Exception error) {
            modelReady = false;
            modelText.setText("Model: unavailable");
            status("Model error");
            log("Model error: " + error.getMessage());
        }
    }

    private MappedByteBuffer loadMappedAsset(String filename) throws IOException {
        AssetFileDescriptor descriptor = getAssets().openFd(filename);
        FileInputStream inputStream = new FileInputStream(descriptor.getFileDescriptor());
        FileChannel channel = inputStream.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.getStartOffset(), descriptor.getDeclaredLength());
    }

    private void toggleCamera() {
        if (cameraRunning) {
            closeCamera();
            return;
        }
        if (!modelReady) {
            log("Built-in model is not ready yet.");
            return;
        }
        if (!hasCameraPermission()) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
            return;
        }
        openCamera();
    }

    private void switchCamera() {
        lensFacing = (lensFacing == CameraCharacteristics.LENS_FACING_BACK) ?
                CameraCharacteristics.LENS_FACING_FRONT : CameraCharacteristics.LENS_FACING_BACK;
        if (cameraRunning) {
            closeCamera();
            openCamera();
        } else {
            log("Camera switched to " + (lensFacing == CameraCharacteristics.LENS_FACING_BACK ? "Back" : "Front"));
        }
    }

    private void openCamera() {
        try {
            CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            cameraId = chooseCamera(manager);
            if (cameraId == null) {
                status("No camera found");
                return;
            }

            imageReader = ImageReader.newInstance(640, 480, ImageFormat.YUV_420_888, 2);
            imageReader.setOnImageAvailableListener(reader -> {
                Image image = reader.acquireLatestImage();
                if (image == null) {
                    return;
                }
                handleFrame(image);
            }, cameraHandler);

            if (!previewView.isAvailable()) {
                status("Preview loading");
                return;
            }

            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                manager.openCamera(cameraId, cameraStateCallback, cameraHandler);
                status("Starting camera");
            }
        } catch (CameraAccessException error) {
            status("Camera error");
            log("Camera error: " + error.getMessage());
        }
    }

    private String chooseCamera(CameraManager manager) throws CameraAccessException {
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(id);
            Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
            if (facing != null && facing == lensFacing) {
                return id;
            }
        }
        return manager.getCameraIdList().length > 0 ? manager.getCameraIdList()[0] : null;
    }

    private final CameraDevice.StateCallback cameraStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraDevice = camera;
            startCameraSession();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            closeCamera();
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            status("Camera error");
            closeCamera();
        }
    };

    private void startCameraSession() {
        try {
            SurfaceTexture texture = previewView.getSurfaceTexture();
            if (texture == null || cameraDevice == null || imageReader == null) {
                return;
            }
            texture.setDefaultBufferSize(640, 480);
            Surface previewSurface = new Surface(texture);
            Surface analysisSurface = imageReader.getSurface();

            CaptureRequest.Builder request = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            request.addTarget(previewSurface);
            request.addTarget(analysisSurface);
            request.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            cameraDevice.createCaptureSession(Arrays.asList(previewSurface, analysisSurface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession session) {
                    captureSession = session;
                    try {
                        captureSession.setRepeatingRequest(request.build(), null, cameraHandler);
                        runOnUiThread(() -> {
                            cameraRunning = true;
                            cameraButton.setText("Stop Camera");
                            status("Live scanning");
                            log("Camera started.");
                        });
                    } catch (CameraAccessException error) {
                        log("Preview error: " + error.getMessage());
                    }
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession session) {
                    status("Camera failed");
                }
            }, cameraHandler);
        } catch (CameraAccessException error) {
            status("Camera error");
            log("Session error: " + error.getMessage());
        }
    }

    private void handleFrame(Image image) {
        long now = System.currentTimeMillis();
        if (processingFrame || now - lastInferenceAt < INFERENCE_INTERVAL_MS) {
            image.close();
            return;
        }
        processingFrame = true;
        lastInferenceAt = now;

        try {
            long started = System.nanoTime();
            Bitmap bitmap = yuvToBitmap(image);
            image.close();
            Bitmap scaled = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true);
            ByteBuffer input = bitmapToModelInput(scaled);
            float[][] output = new float[1][LABELS.length];
            interpreter.run(input, output);

            float[] scores = normalizeScores(output[0]);
            int bestIndex = bestIndex(scores);
            lastInferenceMs = (System.nanoTime() - started) / 1_000_000f;
            frameCount++;

            runOnUiThread(() -> updateResult(bestIndex, scores));
        } catch (Exception error) {
            image.close();
            log("Inference error: " + error.getMessage());
        } finally {
            processingFrame = false;
        }
    }

    private Bitmap yuvToBitmap(Image image) {
        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];
        int width = image.getWidth();
        int height = image.getHeight();
        int[] pixels = new int[width * height];
        ByteBuffer yBuffer = yPlane.getBuffer();
        ByteBuffer uBuffer = uPlane.getBuffer();
        ByteBuffer vBuffer = vPlane.getBuffer();
        int yRowStride = yPlane.getRowStride();
        int uvRowStride = uPlane.getRowStride();
        int uvPixelStride = uPlane.getPixelStride();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int yValue = yBuffer.get(y * yRowStride + x) & 0xff;
                int uvIndex = (y / 2) * uvRowStride + (x / 2) * uvPixelStride;
                int uValue = (uBuffer.get(uvIndex) & 0xff) - 128;
                int vValue = (vBuffer.get(uvIndex) & 0xff) - 128;
                int r = clamp((int) (yValue + 1.402f * vValue));
                int g = clamp((int) (yValue - 0.344136f * uValue - 0.714136f * vValue));
                int b = clamp((int) (yValue + 1.772f * uValue));
                pixels[y * width + x] = Color.rgb(r, g, b);
            }
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
    }

    private ByteBuffer bitmapToModelInput(Bitmap bitmap) {
        ByteBuffer input = ByteBuffer.allocateDirect(4 * inputWidth * inputHeight * 3);
        input.order(ByteOrder.nativeOrder());
        int[] pixels = new int[inputWidth * inputHeight];
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight);
        for (int pixel : pixels) {
            input.putFloat(((Color.red(pixel) / 127.5f) - 1.0f));
            input.putFloat(((Color.green(pixel) / 127.5f) - 1.0f));
            input.putFloat(((Color.blue(pixel) / 127.5f) - 1.0f));
        }
        input.rewind();
        return input;
    }

    private float[] normalizeScores(float[] raw) {
        float sum = 0;
        boolean probabilities = true;
        for (float value : raw) {
            sum += value;
            probabilities = probabilities && value >= 0 && value <= 1;
        }
        if (probabilities && sum > 0.98f && sum < 1.02f) {
            return raw;
        }

        float max = raw[0];
        for (float value : raw) {
            if (value > max) {
                max = value;
            }
        }
        float expSum = 0;
        float[] result = new float[raw.length];
        for (int i = 0; i < raw.length; i++) {
            result[i] = (float) Math.exp(raw[i] - max);
            expSum += result[i];
        }
        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] / expSum;
        }
        return result;
    }

    private int bestIndex(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    private void updateResult(int bestIndex, float[] scores) {
        String emotion = LABELS[bestIndex];
        float confidence = scores[bestIndex];
        emotionText.setText(emotion.toUpperCase(Locale.US));
        emotionText.setTextColor(COLORS[bestIndex]);
        confidenceText.setText(String.format(Locale.US, "%.1f%% confidence", confidence * 100f));
        statsText.setText(String.format(Locale.US, "Frames %d   Inference %.0f ms   Input %dx%d",
                frameCount, lastInferenceMs, inputWidth, inputHeight));

        for (int i = 0; i < LABELS.length; i++) {
            int value = Math.max(0, Math.min(1000, Math.round(scores[i] * 1000f)));
            scoreBars[i].setProgress(value);
            scoreLabels[i].setText(String.format(Locale.US, "%s %.1f%%", LABELS[i], scores[i] * 100f));
        }
        maybeSpeak(emotion, confidence);
        maybePlaySound(emotion, confidence);
    }

    private void initSoundPool() {
        AudioAttributes attrs = new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build();
        soundPool = new SoundPool.Builder()
                .setMaxStreams(1)
                .setAudioAttributes(attrs)
                .build();
        
        soundIds[0] = soundPool.load(this, getResources().getIdentifier("happy_bark", "raw", getPackageName()), 1);
        soundIds[1] = soundPool.load(this, getResources().getIdentifier("sad_howl", "raw", getPackageName()), 1);
        soundIds[2] = soundPool.load(this, getResources().getIdentifier("angry_bark", "raw", getPackageName()), 1);
    }

    private void maybePlaySound(String emotion, float confidence) {
        long now = System.currentTimeMillis();
        if (confidence < 0.45f || now - lastSoundAt < 3500) return;
        
        int index = -1;
        if (emotion.equalsIgnoreCase("Happy")) index = 0;
        else if (emotion.equalsIgnoreCase("Sad")) index = 1;
        else if (emotion.equalsIgnoreCase("Angry")) index = 2;

        if (index != -1 && soundIds[index] != 0) {
            Log.d("DogScanner", "Playing sound via SoundPool: " + emotion);
            soundPool.play(soundIds[index], 1.0f, 1.0f, 1, 0, 1.0f);
            lastSoundAt = now;
            lastPlayedEmotion = emotion;
            runOnUiThread(() -> status("Sensing " + emotion.toUpperCase()));
        }
    }

    private void maybeSpeak(String emotion, float confidence) {
        long now = System.currentTimeMillis();
        if (!voiceEnabled || tts == null || confidence < SPEECH_CONFIDENCE) {
            return;
        }
        if (emotion.equals(lastSpokenEmotion) || now - lastSpeechAt < 3500) {
            return;
        }
        tts.speak(emotion, TextToSpeech.QUEUE_FLUSH, null, "emotion-" + now);
        lastSpokenEmotion = emotion;
        lastSpeechAt = now;
    }

    private void closeCamera() {
        try {
            if (captureSession != null) {
                captureSession.close();
                captureSession = null;
            }
            if (cameraDevice != null) {
                cameraDevice.close();
                cameraDevice = null;
            }
            if (imageReader != null) {
                imageReader.close();
                imageReader = null;
            }
        } finally {
            cameraRunning = false;
            runOnUiThread(() -> {
                cameraButton.setText("Start Camera");
                status(modelReady ? "Model ready" : "Model error");
                log("Camera stopped.");
            });
        }
    }

    private final TextureView.SurfaceTextureListener surfaceListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            if (cameraRunning) {
                openCamera();
            }
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            closeCamera();
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            status("Camera ready");
        } else if (requestCode == CAMERA_REQUEST_CODE) {
            status("Camera denied");
        }
    }

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void startBackgroundThread() {
        cameraThread = new HandlerThread("DogEmotionCamera");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
    }

    private void stopBackgroundThread() {
        if (cameraThread != null) {
            cameraThread.quitSafely();
            try {
                cameraThread.join();
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            cameraThread = null;
            cameraHandler = null;
        }
    }

    @Override
    protected void onDestroy() {
        closeCamera();
        if (interpreter != null) {
            interpreter.close();
        }
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        if (soundPool != null) {
            soundPool.release();
            soundPool = null;
        }
        stopBackgroundThread();
        super.onDestroy();
    }

    private void status(String value) {
        runOnUiThread(() -> {
            statusText.setText(value.toUpperCase(Locale.US));
            statusText.setBackground(roundedRect(0xffe8f5e9, 16));
        });
    }

    private void log(String message) {
        runOnUiThread(() -> logText.setText(message));
    }

    private Button styledButton(String text, int color) {
        Button button = new Button(this);
        button.setText(text);
        button.setTextColor(Color.WHITE);
        button.setTypeface(null, 1);
        button.setBackground(roundedRect(color, 8));
        return button;
    }

    private GradientDrawable roundedRect(int color, int radiusDp) {
        GradientDrawable shape = new GradientDrawable();
        shape.setShape(GradientDrawable.RECTANGLE);
        shape.setCornerRadius(dp(radiusDp));
        shape.setColor(color);
        return shape;
    }

    private TextView cardText(LinearLayout root, String text, int size, int color, int gravity) {
        LinearLayout panel = panel(root);
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(size);
        view.setTextColor(color);
        view.setTypeface(null, 1);
        view.setGravity(gravity);
        panel.addView(view, matchWrap());
        return view;
    }

    private LinearLayout panel(LinearLayout root) {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(16), dp(16), dp(16), dp(16));
        
        GradientDrawable bg = new GradientDrawable();
        bg.setShape(GradientDrawable.RECTANGLE);
        bg.setCornerRadius(dp(12));
        bg.setColor(Color.WHITE);
        bg.setStroke(dp(1), 0xfff0f0f0);
        
        panel.setBackground(bg);
        root.addView(panel, withTopMargin(dp(16)));
        return panel;
    }

    private TextView pill(String text) {
        TextView view = new TextView(this);
        view.setText(text.toUpperCase(Locale.US));
        view.setTextSize(11);
        view.setTypeface(null, 1);
        view.setTextColor(0xff2e7d32);
        view.setGravity(Gravity.CENTER);
        view.setPadding(dp(12), dp(6), dp(12), dp(6));
        view.setBackground(roundedRect(0xffe8f5e9, 16));
        return view;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
    }

    private LinearLayout.LayoutParams withTopMargin(int margin) {
        LinearLayout.LayoutParams params = matchWrap();
        params.topMargin = margin;
        return params;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
