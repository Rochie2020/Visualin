import 'package:camera/camera.dart';
import 'package:jni/jni.dart';

/// Runs synchronously beside hand detection in the existing background isolate.
class UpperBodyDetector {
  static final _class = JClass.forName('com/visualin/app/UpperBodyDetector');
  static final _constructor =
      _class.constructorId('(Landroid/content/Context;)V');
  static final _detect = _class.instanceMethodId('detect',
      '(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIIIIIII)[D');
  static final _close = _class.instanceMethodId('close', '()V');
  final JObject _instance;
  UpperBodyDetector()
      : _instance = _constructor
            .call(_class, JObject.type, [Jni.androidApplicationContext]);

  List<double> detect(CameraImage image, int rotation) {
    final buffers = <JByteBuffer>[];
    try {
      for (final plane in image.planes) {
        buffers.add(JByteBuffer.fromList(plane.bytes));
      }
      final result = _detect.call(_instance, JDoubleArray.type, [
        ...buffers,
        for (final value in [
          image.width,
          image.height,
          ...image.planes.map((p) => p.bytesPerRow),
          image.planes[1].bytesPerPixel ?? 1,
          image.planes[2].bytesPerPixel ?? 1,
          rotation
        ])
          JValueInt(value),
      ]);
      try {
        return result.toList();
      } finally {
        result.release();
      }
    } finally {
      for (final buffer in buffers) {
        buffer.release();
      }
    }
  }

  void dispose() {
    try {
      _close.call(_instance, jvoid.type, []);
    } finally {
      _instance.release();
    }
  }
}
