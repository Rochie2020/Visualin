import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:jni/jni.dart';

/// Packed MediaPipe hand + pose detector using VIDEO tracking, direct ByteBuffer
/// reuse, and zero Bitmap/JPEG round-trips.
class FastLandmarkDetector {
  static final _class =
      JClass.forName('com/visualin/app/FastLandmarkDetector');
  static final _constructor =
      _class.constructorId('(Landroid/content/Context;)V');
  static final _detect = _class.instanceMethodId(
    'detect',
    '(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIIIIIIIIIII)[D',
  );
  static final _releasePose = _class.instanceMethodId('releasePose', '()V');
  static final _close = _class.instanceMethodId('close', '()V');

  final JObject _instance;
  final List<JByteBuffer?> _planeBuffers = [null, null, null];
  final List<Uint8List?> _planeViews = [null, null, null];
  final List<int> _planeCaps = [0, 0, 0];

  FastLandmarkDetector()
      : _instance = _constructor
            .call(_class, JObject.type, [Jni.androidApplicationContext]);

  JByteBuffer _ensurePlaneBuffer(int index, Uint8List bytes) {
    final len = bytes.length;
    var buf = _planeBuffers[index];
    var view = _planeViews[index];
    if (buf == null || view == null || _planeCaps[index] < len) {
      buf?.release();
      buf = JByteBuffer.allocateDirect(len);
      view = buf.asUint8List();
      _planeBuffers[index] = buf;
      _planeViews[index] = view;
      _planeCaps[index] = len;
    }
    view.setRange(0, len, bytes);
    return buf;
  }

  ({
    bool poseSampled,
    bool poseError,
    List<double> pose,
    List<List<List<double>>> hands,
  }) detect(CameraImage image, int rotation, {required bool runPose}) {
    final planes = image.planes;
    final yBuf = _ensurePlaneBuffer(0, planes[0].bytes);
    final uBuf = _ensurePlaneBuffer(1, planes[1].bytes);
    final vBuf = _ensurePlaneBuffer(2, planes[2].bytes);

    final packed = _detect.call(_instance, JDoubleArray.type, [
      yBuf,
      uBuf,
      vBuf,
      for (final value in [
        planes[0].bytes.length,
        planes[1].bytes.length,
        planes[2].bytes.length,
        image.width,
        image.height,
        planes[0].bytesPerRow,
        planes[1].bytesPerRow,
        planes[2].bytesPerRow,
        planes[1].bytesPerPixel ?? 1,
        planes[2].bytesPerPixel ?? 1,
        rotation,
        runPose ? 1 : 0,
      ])
        JValueInt(value),
    ]);
    try {
      final values = packed.toList();
      if (values.length < 4) {
        return (
          poseSampled: runPose,
          poseError: false,
          pose: const [],
          hands: const [],
        );
      }
      final poseSampled = values[0] >= 0.5;
      final poseError = values[1] >= 0.5;
      final poseCount = values[2].toInt();
      var cursor = 3;
      final pose = poseCount > 0 && cursor + poseCount * 2 <= values.length
          ? values.sublist(cursor, cursor + poseCount * 2)
          : <double>[];
      cursor += poseCount * 2;
      final hands = <List<List<double>>>[];
      if (cursor < values.length) {
        final handCount = values[cursor++].toInt();
        for (var h = 0; h < handCount; h++) {
          if (cursor + 63 > values.length) break;
          hands.add([
            for (var i = 0; i < 21; i++)
              <double>[
                values[cursor + i * 3],
                values[cursor + i * 3 + 1],
                values[cursor + i * 3 + 2],
              ],
          ]);
          cursor += 63;
        }
      }
      return (
        poseSampled: poseSampled,
        poseError: poseError,
        pose: pose,
        hands: hands,
      );
    } finally {
      packed.release();
    }
  }

  void releasePose() {
    _releasePose.call(_instance, jvoid.type, []);
  }

  void dispose() {
    for (var i = 0; i < _planeBuffers.length; i++) {
      _planeBuffers[i]?.release();
      _planeBuffers[i] = null;
      _planeViews[i] = null;
      _planeCaps[i] = 0;
    }
    try {
      _close.call(_instance, jvoid.type, []);
    } finally {
      _instance.release();
    }
  }
}
