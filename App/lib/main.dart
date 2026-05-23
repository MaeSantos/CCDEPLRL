import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const DogEmotionApp());
}

class DogEmotionApp extends StatelessWidget {
  const DogEmotionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Dog Emotion',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xffe67b44),
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xfff4f7f5),
        useMaterial3: true,
        fontFamily: 'Roboto',
      ),
      home: const ScannerPage(),
    );
  }
}

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> {
  static const labels = ['Angry', 'Happy', 'Sad'];
  static const emojis = {'Angry': '😠', 'Happy': '😄', 'Sad': '😢'};
  static const speechConfidence = 0.65;
  static const stableFrames = 3;
  static const speechCooldown = Duration(milliseconds: 3500);

  final _tts = FlutterTts();
  final _log = <String>[];

  CameraController? _camera;
  Interpreter? _interpreter;
  List<CameraDescription> _cameras = [];
  int _cameraIndex = 0;
  int _inputWidth = 224;
  int _inputHeight = 224;
  int _frameCount = 0;
  double _inferenceMs = 0;
  DateTime _lastInference = DateTime.fromMillisecondsSinceEpoch(0);
  bool _isLoadingModel = false;
  bool _isStartingCamera = false;
  bool _isInferencing = false;
  bool _voiceEnabled = true;
  String _status = 'Load a TFLite model';
  String? _modelName;
  String _emotion = 'Waiting';
  double _confidence = 0;
  List<double> _scores = [0, 0, 0];
  String? _candidateEmotion;
  int _candidateCount = 0;
  String? _lastSpokenEmotion;
  DateTime _lastSpeechAt = DateTime.fromMillisecondsSinceEpoch(0);

  bool get _modelReady => _interpreter != null;
  bool get _cameraReady => _camera?.value.isInitialized ?? false;
  bool get _scanning =>
      _cameraReady && (_camera?.value.isStreamingImages ?? false);

  @override
  void initState() {
    super.initState();
    _configureTts();
    _discoverCameras();
    _addLog('Native Flutter scanner ready.');
  }

  Future<void> _configureTts() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.45);
    await _tts.setVolume(1);
  }

  Future<void> _discoverCameras() async {
    try {
      _cameras = await availableCameras();
      if (mounted) {
        setState(
          () => _status = _cameras.isEmpty ? 'No camera found' : 'Camera ready',
        );
      }
    } catch (err) {
      _addLog('Camera lookup failed: $err');
      if (mounted) {
        setState(() => _status = 'Camera unavailable');
      }
    }
  }

  @override
  void dispose() {
    _camera?.dispose();
    _interpreter?.close();
    _tts.stop();
    super.dispose();
  }

  Future<void> _pickModel() async {
    if (_isLoadingModel) return;
    setState(() {
      _isLoadingModel = true;
      _status = 'Loading model...';
    });

    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['tflite'],
        withData: true,
      );
      if (result == null || result.files.isEmpty) {
        setState(() {
          _isLoadingModel = false;
          _status = _modelReady ? 'Model ready' : 'Load a TFLite model';
        });
        return;
      }

      final file = result.files.single;
      final modelFile = await _materializePickedFile(file);
      final nextInterpreter = Interpreter.fromFile(modelFile);
      final inputShape = nextInterpreter.getInputTensor(0).shape;
      final outputShape = nextInterpreter.getOutputTensor(0).shape;

      if (inputShape.length != 4 || inputShape.last != 3) {
        nextInterpreter.close();
        throw Exception(
          'Expected input shape [1, height, width, 3], got $inputShape',
        );
      }
      if (outputShape.isEmpty || outputShape.last != labels.length) {
        nextInterpreter.close();
        throw Exception(
          'Expected ${labels.length} output scores, got $outputShape',
        );
      }

      _interpreter?.close();
      _interpreter = nextInterpreter;
      _inputHeight = inputShape[1];
      _inputWidth = inputShape[2];

      setState(() {
        _modelName = file.name;
        _isLoadingModel = false;
        _status = 'Model ready';
      });
      _addLog('Loaded ${file.name} at ${_inputWidth}x$_inputHeight.');
    } catch (err) {
      setState(() {
        _isLoadingModel = false;
        _status = 'Model load failed';
      });
      _addLog('Model error: $err');
    }
  }

  Future<File> _materializePickedFile(PlatformFile file) async {
    if (file.path != null) {
      return File(file.path!);
    }
    final bytes = file.bytes;
    if (bytes == null) {
      throw Exception('File picker returned no readable model data.');
    }
    final dir = await getApplicationDocumentsDirectory();
    final safeName = file.name.replaceAll(RegExp(r'[^A-Za-z0-9_.-]'), '_');
    final target = File('${dir.path}/$safeName');
    return target.writeAsBytes(bytes, flush: true);
  }

  Future<void> _startCamera() async {
    if (!_modelReady) {
      _addLog('Load a .tflite model first.');
      return;
    }
    if (_cameras.isEmpty || _isStartingCamera) return;

    setState(() {
      _isStartingCamera = true;
      _status = 'Starting camera...';
    });

    try {
      await _camera?.dispose();
      final controller = CameraController(
        _cameras[_cameraIndex],
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      await controller.startImageStream(_onFrame);

      setState(() {
        _camera = controller;
        _frameCount = 0;
        _status = 'Live scanning';
        _isStartingCamera = false;
      });
      _addLog('Camera started.');
    } catch (err) {
      setState(() {
        _isStartingCamera = false;
        _status = 'Camera error';
      });
      _addLog('Camera error: $err');
    }
  }

  Future<void> _stopCamera() async {
    final controller = _camera;
    if (controller == null) return;
    if (controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
    await controller.dispose();
    setState(() {
      _camera = null;
      _status = _modelReady ? 'Model ready' : 'Load a TFLite model';
      _emotion = 'Waiting';
      _confidence = 0;
      _scores = [0, 0, 0];
      _frameCount = 0;
      _candidateEmotion = null;
      _candidateCount = 0;
      _lastSpokenEmotion = null;
    });
    _addLog('Camera stopped.');
  }

  Future<void> _flipCamera() async {
    if (_cameras.length < 2) return;
    final wasScanning = _scanning;
    await _stopCamera();
    _cameraIndex = (_cameraIndex + 1) % _cameras.length;
    if (wasScanning) {
      await _startCamera();
    }
  }

  Future<void> _onFrame(CameraImage frame) async {
    final now = DateTime.now();
    if (_isInferencing || now.difference(_lastInference).inMilliseconds < 180) {
      return;
    }
    _lastInference = now;
    _isInferencing = true;

    final sw = Stopwatch()..start();
    try {
      final interpreter = _interpreter;
      if (interpreter == null) return;

      final rgb = _cameraImageToImage(frame);
      final resized = img.copyResize(
        rgb,
        width: _inputWidth,
        height: _inputHeight,
      );
      final input = _imageToModelInput(resized);
      final output = List.generate(
        1,
        (_) => List<double>.filled(labels.length, 0),
      );

      interpreter.run(input, output);
      final normalized = _normalizeScores(output.first);
      final bestIndex = _bestIndex(normalized);
      final emotion = labels[bestIndex];
      final confidence = normalized[bestIndex];

      sw.stop();
      if (!mounted) return;
      setState(() {
        _scores = normalized;
        _emotion = emotion;
        _confidence = confidence;
        _inferenceMs = sw.elapsedMicroseconds / 1000;
        _frameCount++;
      });
      _maybeSpeak(emotion, confidence);
    } catch (err) {
      _addLog('Inference error: $err');
    } finally {
      _isInferencing = false;
    }
  }

  List<List<List<List<double>>>> _imageToModelInput(img.Image image) {
    return [
      List.generate(_inputHeight, (y) {
        return List.generate(_inputWidth, (x) {
          final pixel = image.getPixel(x, y);
          return [
            pixel.r / 127.5 - 1,
            pixel.g / 127.5 - 1,
            pixel.b / 127.5 - 1,
          ];
        });
      }),
    ];
  }

  img.Image _cameraImageToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final out = img.Image(width: width, height: height);
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
        final yp = yPlane.bytes[y * yPlane.bytesPerRow + x];
        final up = uPlane.bytes[uvIndex];
        final vp = vPlane.bytes[uvIndex];

        final yf = yp.toDouble();
        final uf = up.toDouble() - 128;
        final vf = vp.toDouble() - 128;
        final r = (yf + 1.402 * vf).round().clamp(0, 255);
        final g = (yf - 0.344136 * uf - 0.714136 * vf).round().clamp(0, 255);
        final b = (yf + 1.772 * uf).round().clamp(0, 255);
        out.setPixelRgb(x, y, r, g, b);
      }
    }

    final rotation = _cameras.isEmpty
        ? 0
        : _cameras[_cameraIndex].sensorOrientation;
    if (rotation == 90 || rotation == 270) {
      return img.copyRotate(out, angle: rotation);
    }
    return out;
  }

  List<double> _normalizeScores(List<double> raw) {
    final values = raw.take(labels.length).toList();
    final sum = values.fold<double>(0, (total, value) => total + value);
    final looksLikeProbability =
        values.every((v) => v >= 0 && v <= 1) && sum > 0.98 && sum < 1.02;
    if (looksLikeProbability) return values;

    final maxValue = values.reduce(math.max);
    final exps = values.map((v) => math.exp(v - maxValue)).toList();
    final expSum = exps.fold<double>(0, (total, value) => total + value);
    return exps.map((v) => v / expSum).toList();
  }

  int _bestIndex(List<double> values) {
    var best = 0;
    for (var i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  Future<void> _maybeSpeak(String emotion, double confidence) async {
    if (!_voiceEnabled || confidence < speechConfidence) {
      _candidateEmotion = null;
      _candidateCount = 0;
      return;
    }

    if (_candidateEmotion == emotion) {
      _candidateCount++;
    } else {
      _candidateEmotion = emotion;
      _candidateCount = 1;
    }

    final now = DateTime.now();
    final stable = _candidateCount >= stableFrames;
    final cooledDown = now.difference(_lastSpeechAt) > speechCooldown;
    final changed = emotion != _lastSpokenEmotion;
    if (!stable || !cooledDown || !changed) return;

    await _tts.stop();
    await _tts.setPitch(
      emotion == 'Happy'
          ? 1.12
          : emotion == 'Sad'
          ? 0.82
          : 0.95,
    );
    await _tts.speak(emotion);
    _lastSpokenEmotion = emotion;
    _lastSpeechAt = now;
  }

  void _addLog(String message) {
    if (!mounted) return;
    setState(() {
      final time = TimeOfDay.now().format(context);
      _log.insert(0, '[$time] $message');
      if (_log.length > 5) {
        _log.removeLast();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final emotionColor = switch (_emotion) {
      'Angry' => const Color(0xffd84a45),
      'Happy' => const Color(0xff249a65),
      'Sad' => const Color(0xff3d73bf),
      _ => const Color(0xffa9bdb3),
    };

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _Header(status: _status, scanning: _scanning),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 10, 16, 18),
                children: [
                  _CameraPanel(
                    camera: _camera,
                    canStart: _modelReady && !_isStartingCamera,
                    isStarting: _isStartingCamera,
                    scanning: _scanning,
                    onStart: _startCamera,
                    onStop: _stopCamera,
                    onFlip: _cameras.length > 1 ? _flipCamera : null,
                  ),
                  const SizedBox(height: 14),
                  _EmotionCard(
                    emotion: _emotion,
                    emoji: emojis[_emotion] ?? '🐾',
                    confidence: _confidence,
                    color: emotionColor,
                  ),
                  const SizedBox(height: 14),
                  _ModelCard(
                    modelName: _modelName,
                    loading: _isLoadingModel,
                    onPickModel: _pickModel,
                  ),
                  const SizedBox(height: 14),
                  _VoiceCard(
                    enabled: _voiceEnabled,
                    onChanged: (value) async {
                      setState(() => _voiceEnabled = value);
                      if (!value) {
                        await _tts.stop();
                      }
                    },
                  ),
                  const SizedBox(height: 14),
                  _ScoresCard(scores: _scores),
                  const SizedBox(height: 14),
                  _StatsCard(
                    frames: _frameCount,
                    inferenceMs: _inferenceMs,
                    inputSize: '${_inputWidth}x$_inputHeight',
                  ),
                  const SizedBox(height: 14),
                  _LogCard(log: _log),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.status, required this.scanning});

  final String status;
  final bool scanning;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xffdce8e2)),
              boxShadow: const [
                BoxShadow(
                  color: Color(0x1726322e),
                  blurRadius: 18,
                  offset: Offset(0, 8),
                ),
              ],
            ),
            child: const Center(
              child: Text('🐶', style: TextStyle(fontSize: 24)),
            ),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Dog Emotion',
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800),
                ),
                Text(
                  'Native scanner',
                  style: TextStyle(
                    color: Color(0xff64756d),
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
          Chip(
            label: Text(status),
            avatar: Icon(scanning ? Icons.sensors : Icons.circle, size: 14),
            side: BorderSide.none,
            backgroundColor: scanning ? const Color(0xffebfff5) : Colors.white,
            labelStyle: TextStyle(
              color: scanning
                  ? const Color(0xff249a65)
                  : const Color(0xff64756d),
              fontWeight: FontWeight.w800,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

class _CameraPanel extends StatelessWidget {
  const _CameraPanel({
    required this.camera,
    required this.canStart,
    required this.isStarting,
    required this.scanning,
    required this.onStart,
    required this.onStop,
    required this.onFlip,
  });

  final CameraController? camera;
  final bool canStart;
  final bool isStarting;
  final bool scanning;
  final VoidCallback onStart;
  final VoidCallback onStop;
  final VoidCallback? onFlip;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      padding: EdgeInsets.zero,
      child: Column(
        children: [
          AspectRatio(
            aspectRatio: 4 / 5,
            child: ClipRRect(
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(8),
              ),
              child: ColoredBox(
                color: const Color(0xff18211d),
                child: camera != null && camera!.value.isInitialized
                    ? CameraPreview(camera!)
                    : const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.photo_camera_outlined,
                              color: Colors.white70,
                              size: 54,
                            ),
                            SizedBox(height: 12),
                            Text(
                              'Camera preview will appear here',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                      ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: scanning
                        ? onStop
                        : canStart
                        ? onStart
                        : null,
                    icon: Icon(scanning ? Icons.stop : Icons.play_arrow),
                    label: Text(
                      isStarting
                          ? 'Starting...'
                          : scanning
                          ? 'Stop'
                          : 'Start Camera',
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filledTonal(
                  onPressed: scanning ? onFlip : null,
                  icon: const Icon(Icons.cameraswitch),
                  tooltip: 'Flip camera',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _EmotionCard extends StatelessWidget {
  const _EmotionCard({
    required this.emotion,
    required this.emoji,
    required this.confidence,
    required this.color,
  });

  final String emotion;
  final String emoji;
  final double confidence;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Column(
        children: [
          Text(emoji, style: const TextStyle(fontSize: 68)),
          Text(
            emotion.toUpperCase(),
            style: TextStyle(
              fontSize: 34,
              height: 1,
              fontWeight: FontWeight.w900,
              color: color,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            confidence == 0
                ? 'Load model, start camera, aim at dog'
                : '${(confidence * 100).toStringAsFixed(1)}% confidence',
            style: const TextStyle(
              color: Color(0xff64756d),
              fontWeight: FontWeight.w700,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

class _ModelCard extends StatelessWidget {
  const _ModelCard({
    required this.modelName,
    required this.loading,
    required this.onPickModel,
  });

  final String? modelName;
  final bool loading;
  final VoidCallback onPickModel;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _SectionTitle('Model'),
          OutlinedButton.icon(
            onPressed: loading ? null : onPickModel,
            icon: const Icon(Icons.upload_file),
            label: Text(
              loading ? 'Loading model...' : modelName ?? 'Pick .tflite model',
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            'Use a TFLite model with output order: Angry, Happy, Sad.',
            style: TextStyle(
              color: Color(0xff64756d),
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _VoiceCard extends StatelessWidget {
  const _VoiceCard({required this.enabled, required this.onChanged});

  final bool enabled;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Row(
        children: [
          const Icon(Icons.record_voice_over, color: Color(0xff249a65)),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Voice feedback',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900),
                ),
                Text(
                  'Speaks Happy, Sad, or Angry.',
                  style: TextStyle(color: Color(0xff64756d)),
                ),
              ],
            ),
          ),
          Switch(value: enabled, onChanged: onChanged),
        ],
      ),
    );
  }
}

class _ScoresCard extends StatelessWidget {
  const _ScoresCard({required this.scores});

  final List<double> scores;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _SectionTitle('Confidence'),
          _ScoreBar(
            label: 'Angry',
            icon: '😠',
            value: scores[0],
            color: const Color(0xffd84a45),
          ),
          _ScoreBar(
            label: 'Happy',
            icon: '😄',
            value: scores[1],
            color: const Color(0xff249a65),
          ),
          _ScoreBar(
            label: 'Sad',
            icon: '😢',
            value: scores[2],
            color: const Color(0xff3d73bf),
          ),
        ],
      ),
    );
  }
}

class _ScoreBar extends StatelessWidget {
  const _ScoreBar({
    required this.label,
    required this.icon,
    required this.value,
    required this.color,
  });

  final String label;
  final String icon;
  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  '$icon $label',
                  style: TextStyle(color: color, fontWeight: FontWeight.w900),
                ),
              ),
              Text(
                '${(value * 100).toStringAsFixed(1)}%',
                style: TextStyle(color: color, fontWeight: FontWeight.w900),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              value: value.clamp(0, 1),
              minHeight: 10,
              color: color,
              backgroundColor: const Color(0xffe6eee9),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatsCard extends StatelessWidget {
  const _StatsCard({
    required this.frames,
    required this.inferenceMs,
    required this.inputSize,
  });

  final int frames;
  final double inferenceMs;
  final String inputSize;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Row(
        children: [
          _Stat(label: 'Frames', value: '$frames'),
          _Stat(
            label: 'Inference',
            value: inferenceMs == 0
                ? '-'
                : '${inferenceMs.toStringAsFixed(0)} ms',
          ),
          _Stat(label: 'Input', value: inputSize),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w900,
              color: Color(0xffe67b44),
            ),
          ),
          Text(
            label,
            style: const TextStyle(
              color: Color(0xff64756d),
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _LogCard extends StatelessWidget {
  const _LogCard({required this.log});

  final List<String> log;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _SectionTitle('System log'),
          if (log.isEmpty)
            const Text(
              'No events yet.',
              style: TextStyle(color: Color(0xff64756d)),
            )
          else
            ...log.map(
              (line) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Text(
                  line,
                  style: const TextStyle(
                    color: Color(0xff64756d),
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _Panel extends StatelessWidget {
  const _Panel({required this.child, this.padding = const EdgeInsets.all(16)});

  final Widget child;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xffdce8e2)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x1426322e),
            blurRadius: 24,
            offset: Offset(0, 10),
          ),
        ],
      ),
      child: child,
    );
  }
}

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        text,
        style: const TextStyle(
          color: Color(0xff64756d),
          fontWeight: FontWeight.w900,
          fontSize: 13,
        ),
      ),
    );
  }
}
