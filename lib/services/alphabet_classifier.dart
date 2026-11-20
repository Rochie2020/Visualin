import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';

import 'landmark_service.dart';
import 'sign_classifier.dart';

/// On-device static FSL alphabet classifier powered by the exported `.tflite`
/// MLP model and matching 88-feature MediaPipe normalization.
class AlphabetClassifier {
  AlphabetClassifier._({
    required this.labels,
    required this.threshold,
    required this.inputFeatures,
    required List<_TfliteOp> ops,
    required int inputTensorIndex,
    required int outputTensorIndex,
    required List<Float32List?> tensorBuffers,
    required List<List<int>> tensorShapes,
  })  : _ops = ops,
        _inputTensorIndex = inputTensorIndex,
        _outputTensorIndex = outputTensorIndex,
        _tensorBuffers = tensorBuffers,
        _tensorShapes = tensorShapes;

  static const int featureCount = 88;
  static const List<int> _thumbToFingerTargets = <int>[
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    17,
    20,
  ];
  static const List<int> _fingertips = <int>[4, 8, 12, 16, 20];
  static const List<(int, int)> _adjacentTipPairs = <(int, int)>[
    (8, 12),
    (12, 16),
    (16, 20),
    (4, 16),
  ];
  static const List<(int, int, int, int)> _curlChains = <(int, int, int, int)>[
    (1, 2, 2, 4),
    (5, 6, 6, 8),
    (9, 10, 10, 12),
    (13, 14, 14, 16),
    (17, 18, 18, 20),
  ];

  final List<String> labels;
  final double threshold;
  final int inputFeatures;
  final List<_TfliteOp> _ops;
  final int _inputTensorIndex;
  final int _outputTensorIndex;
  final List<Float32List?> _tensorBuffers;
  final List<List<int>> _tensorShapes;

  static Future<AlphabetClassifier> loadFromAssets({
    String modelAsset = 'assets/ml/alphabet_classifier.tflite',
    String configAsset = 'assets/ml/alphabet_preprocessing.json',
  }) async {
    final configText = await rootBundle.loadString(configAsset);
    final config = jsonDecode(configText) as Map<String, dynamic>;
    final labels = List<String>.from(config['labels'] as List);
    final threshold = (config['threshold'] as num?)?.toDouble() ?? 0.70;
    final modelData = await rootBundle.load(modelAsset);
    final bytes = modelData.buffer.asUint8List(
      modelData.offsetInBytes,
      modelData.lengthInBytes,
    );
    return AlphabetClassifier.fromTfliteBytes(
      bytes,
      labels: labels,
      threshold: threshold,
    );
  }

  /// Parses the exported float32 `.tflite` FlatBuffer (`FULLY_CONNECTED` +
  /// `SOFTMAX`) directly so inference matches `tf.lite.Interpreter` without an
  /// external native runtime plugin.
  factory AlphabetClassifier.fromTfliteBytes(
    Uint8List tfliteBytes, {
    required List<String> labels,
    double threshold = 0.70,
  }) {
    if (labels.isEmpty) {
      throw ArgumentError('Alphabet labels must not be empty');
    }
    if (tfliteBytes.length < 16 ||
        String.fromCharCodes(tfliteBytes.sublist(4, 8)) != 'TFL3') {
      throw const FormatException('Invalid TFLite FlatBuffer header');
    }

    final fb = _FlatBufferReader(tfliteBytes);
    final root = fb.u32(0);

    // OperatorCode table: slot 0 = deprecated_builtin_code (int8),
    // slot 3 = builtin_code (int32).
    final opcodesVec = fb.field(root, 1);
    final opcodesCount = fb.vecLen(opcodesVec);
    final builtinCodes = <int>[];
    for (var i = 0; i < opcodesCount; i++) {
      final oc = fb.vecTableAt(opcodesVec, i);
      final b3Field = fb.field(oc, 3);
      final b0Field = fb.field(oc, 0);
      final code = b3Field != 0
          ? fb.i32(b3Field)
          : (b0Field != 0 ? fb.u8(b0Field) : 0);
      builtinCodes.add(code);
    }

    // Buffers table vector (Model field 4).
    final buffersVec = fb.field(root, 4);
    final buffersCount = fb.vecLen(buffersVec);
    final rawBuffers = <Uint8List?>[];
    for (var i = 0; i < buffersCount; i++) {
      final bTable = fb.vecTableAt(buffersVec, i);
      final dataField = fb.field(bTable, 0);
      if (dataField == 0) {
        rawBuffers.add(null);
      } else {
        rawBuffers.add(fb.byteVector(dataField));
      }
    }

    // Subgraphs (Model field 2), use primary subgraph 0.
    final subgraphsVec = fb.field(root, 2);
    if (fb.vecLen(subgraphsVec) == 0) {
      throw const FormatException('TFLite model has no subgraphs');
    }
    final subgraph = fb.vecTableAt(subgraphsVec, 0);

    final tensorsVec = fb.field(subgraph, 0);
    final tensorsCount = fb.vecLen(tensorsVec);
    final tensorShapes = <List<int>>[];
    final tensorBuffers = <Float32List?>[];
    for (var i = 0; i < tensorsCount; i++) {
      final t = fb.vecTableAt(tensorsVec, i);
      final shapeField = fb.field(t, 0);
      final shape = shapeField != 0 ? fb.i32Vector(shapeField) : <int>[];
      final typeField = fb.field(t, 1);
      final dtype = typeField != 0 ? fb.u8(typeField) : 0;
      if (dtype != 0) {
        throw FormatException('Unsupported TFLite tensor dtype: $dtype');
      }
      final bufField = fb.field(t, 2);
      final bufIdx = bufField != 0 ? fb.u32(bufField) : 0;
      final raw = bufIdx < rawBuffers.length ? rawBuffers[bufIdx] : null;
      Float32List? floats;
      if (raw != null && raw.isNotEmpty) {
        if (raw.length % 4 != 0) {
          throw const FormatException('Misaligned float32 buffer in TFLite');
        }
        final count = raw.length ~/ 4;
        floats = Float32List(count);
        final bd = ByteData.sublistView(raw);
        for (var k = 0; k < count; k++) {
          floats[k] = bd.getFloat32(k * 4, Endian.little);
        }
      }
      tensorShapes.add(shape);
      tensorBuffers.add(floats);
    }

    final inputsField = fb.field(subgraph, 1);
    final outputsField = fb.field(subgraph, 2);
    final inputIds = fb.i32Vector(inputsField);
    final outputIds = fb.i32Vector(outputsField);
    if (inputIds.length != 1 || outputIds.length != 1) {
      throw const FormatException('Expected single-input, single-output TFLite');
    }
    final inputTensorIndex = inputIds.first;
    final outputTensorIndex = outputIds.first;
    final inShape = tensorShapes[inputTensorIndex];
    final outShape = tensorShapes[outputTensorIndex];
    if (inShape.length != 2 ||
        inShape[0] != 1 ||
        inShape[1] != featureCount ||
        outShape.length != 2 ||
        outShape[0] != 1 ||
        outShape[1] != labels.length) {
      throw FormatException(
        'Unexpected TFLite I/O shapes: in=$inShape, out=$outShape, labels=${labels.length}',
      );
    }

    final opsVec = fb.field(subgraph, 3);
    final opsCount = fb.vecLen(opsVec);
    final ops = <_TfliteOp>[];
    for (var i = 0; i < opsCount; i++) {
      final op = fb.vecTableAt(opsVec, i);
      final opcodeField = fb.field(op, 0);
      final opcodeIdx = opcodeField != 0 ? fb.u32(opcodeField) : 0;
      final builtinCode = builtinCodes[opcodeIdx];
      final opInputs = fb.i32Vector(fb.field(op, 1));
      final opOutputs = fb.i32Vector(fb.field(op, 2));
      final optsField = fb.field(op, 4);
      final optsTable = optsField != 0 ? fb.indirect(optsField) : 0;

      if (builtinCode == 9) {
        // FULLY_CONNECTED: inputs = [input, weights, bias?], outputs = [output]
        if (opInputs.length < 2 || opOutputs.length != 1) {
          throw const FormatException('Invalid FULLY_CONNECTED operator');
        }
        final fusedField = optsTable != 0 ? fb.field(optsTable, 0) : 0;
        final fused = fusedField != 0 ? fb.i8(fusedField) : 0;
        if (fused != 0 && fused != 1) {
          throw FormatException('Unsupported fused activation: $fused');
        }
        ops.add(_FullyConnectedOp(
          inputIndex: opInputs[0],
          weightsIndex: opInputs[1],
          biasIndex: opInputs.length >= 3 ? opInputs[2] : -1,
          outputIndex: opOutputs[0],
          relu: fused == 1,
        ));
      } else if (builtinCode == 25) {
        // SOFTMAX: inputs = [input], outputs = [output], options slot 0 = beta
        if (opInputs.length != 1 || opOutputs.length != 1) {
          throw const FormatException('Invalid SOFTMAX operator');
        }
        final betaField = optsTable != 0 ? fb.field(optsTable, 0) : 0;
        final beta = betaField != 0 ? fb.f32(betaField) : 1.0;
        ops.add(_SoftmaxOp(
          inputIndex: opInputs[0],
          outputIndex: opOutputs[0],
          beta: beta,
        ));
      } else {
        throw FormatException('Unsupported TFLite builtin opcode: $builtinCode');
      }
    }

    return AlphabetClassifier._(
      labels: labels,
      threshold: threshold,
      inputFeatures: inShape[1],
      ops: ops,
      inputTensorIndex: inputTensorIndex,
      outputTensorIndex: outputTensorIndex,
      tensorBuffers: tensorBuffers,
      tensorShapes: tensorShapes,
    );
  }

  /// Extracts the 88-dim feature vector from 21 upright 3D hand landmarks in
  /// image-height units on all three axes. Returns `null` if invalid or too small.
  static Float64List? extractFeatures(List<LandmarkPoint> hand) {
    if (hand.length != 21) return null;
    for (final p in hand) {
      if (!p.x.isFinite || !p.y.isFinite || !p.z.isFinite) return null;
    }
    final wrist = hand[0];
    var palmSum = 0.0;
    for (final idx in const [5, 9, 13, 17]) {
      final dx = hand[idx].x - wrist.x;
      final dy = hand[idx].y - wrist.y;
      palmSum += math.sqrt(dx * dx + dy * dy);
    }
    final palm = palmSum / 4.0;
    if (!palm.isFinite || palm <= 1e-4) return null;

    final feats = Float64List(featureCount);
    for (var j = 0; j < 21; j++) {
      feats[j * 3] = ((hand[j].x - wrist.x) / palm).clamp(-5.0, 5.0);
      feats[j * 3 + 1] = ((hand[j].y - wrist.y) / palm).clamp(-5.0, 5.0);
      feats[j * 3 + 2] = ((hand[j].z - wrist.z) / palm).clamp(-5.0, 5.0);
    }

    double dist2(int a, int b) {
      final dx = feats[a * 3] - feats[b * 3];
      final dy = feats[a * 3 + 1] - feats[b * 3 + 1];
      return math.sqrt(dx * dx + dy * dy);
    }

    var cursor = 63;
    for (final j in _thumbToFingerTargets) {
      feats[cursor++] = dist2(4, j);
    }
    for (final j in _fingertips) {
      feats[cursor++] = dist2(0, j);
    }
    for (final (a, b) in _adjacentTipPairs) {
      feats[cursor++] = dist2(a, b);
    }
    for (final (a, b, c, d) in _curlChains) {
      final ux = feats[b * 3] - feats[a * 3];
      final uy = feats[b * 3 + 1] - feats[a * 3 + 1];
      final vx = feats[d * 3] - feats[c * 3];
      final vy = feats[d * 3 + 1] - feats[c * 3 + 1];
      final un = math.sqrt(ux * ux + uy * uy);
      final vn = math.sqrt(vx * vx + vy * vy);
      final denom = math.max(un * vn, 1e-6);
      feats[cursor++] = (ux * vx + uy * vy) / denom;
    }

    final u2x = feats[8 * 3] - feats[12 * 3];
    final u2y = feats[8 * 3 + 1] - feats[12 * 3 + 1];
    final v2x = feats[5 * 3] - feats[9 * 3];
    final v2y = feats[5 * 3 + 1] - feats[9 * 3 + 1];
    final u2n = math.sqrt(u2x * u2x + u2y * u2y);
    final v2n = math.sqrt(v2x * v2x + v2y * v2y);
    final denom2 = math.max(u2n * v2n, 1e-6);
    feats[cursor++] = (u2x * v2x + u2y * v2y) / denom2;

    return feats;
  }

  /// Evaluates the `.tflite` model on an already normalized 88-dim feature vector.
  Float64List predictProbsFromFeatures(List<double> features) {
    if (features.length != inputFeatures) {
      throw ArgumentError(
          'Expected $inputFeatures features, got ${features.length}');
    }
    final activations = List<Float32List?>.generate(
      _tensorBuffers.length,
      (i) => _tensorBuffers[i],
    );
    final inBuf = Float32List(inputFeatures);
    for (var i = 0; i < inputFeatures; i++) {
      inBuf[i] = features[i];
    }
    activations[_inputTensorIndex] = inBuf;

    for (final op in _ops) {
      op.execute(activations, _tensorShapes);
    }

    final out = activations[_outputTensorIndex]!;
    final probs = Float64List(out.length);
    for (var i = 0; i < out.length; i++) {
      probs[i] = out[i];
    }
    return probs;
  }

  SignPrediction? predictHand(List<LandmarkPoint> hand) {
    final feats = extractFeatures(hand);
    if (feats == null) return null;
    final probs = predictProbsFromFeatures(feats);
    var best = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) best = i;
    }
    return SignPrediction(labels[best], probs[best], threshold);
  }

  /// Returns the top [count] predicted `(label, probability)` pairs for [hand].
  List<MapEntry<String, double>> topCandidates(
    List<LandmarkPoint> hand, {
    int count = 3,
  }) {
    final feats = extractFeatures(hand);
    if (feats == null) return const [];
    final probs = predictProbsFromFeatures(feats);
    final indices = List<int>.generate(probs.length, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));
    return [
      for (final idx in indices.take(count)) MapEntry(labels[idx], probs[idx]),
    ];
  }

  /// Selects the primary valid hand (smallest wrist `y`, matching
  /// [SignClassifier.predict]) and predicts its static alphabet letter.
  SignPrediction? predict(List<List<LandmarkPoint>> hands) {
    final valid = <List<LandmarkPoint>>[];
    for (final hand in hands) {
      if (extractFeatures(hand) != null) valid.add(hand);
    }
    if (valid.isEmpty) return null;
    valid.sort((a, b) => a.first.y.compareTo(b.first.y));
    return predictHand(valid.first);
  }

  bool accepts(SignPrediction result) =>
      result.score >= (result.minimumScore ?? threshold);
}

abstract class _TfliteOp {
  void execute(List<Float32List?> tensors, List<List<int>> shapes);
}

class _FullyConnectedOp implements _TfliteOp {
  const _FullyConnectedOp({
    required this.inputIndex,
    required this.weightsIndex,
    required this.biasIndex,
    required this.outputIndex,
    required this.relu,
  });

  final int inputIndex;
  final int weightsIndex;
  final int biasIndex;
  final int outputIndex;
  final bool relu;

  @override
  void execute(List<Float32List?> tensors, List<List<int>> shapes) {
    final x = tensors[inputIndex]!;
    final w = tensors[weightsIndex]!;
    final b = biasIndex >= 0 ? tensors[biasIndex] : null;
    final wShape = shapes[weightsIndex];
    final outDim = wShape[0];
    final inDim = wShape[1];
    final out = Float32List(outDim);
    for (var o = 0; o < outDim; o++) {
      var sum = b != null ? b[o] : 0.0;
      final rowOffset = o * inDim;
      for (var i = 0; i < inDim; i++) {
        sum += x[i] * w[rowOffset + i];
      }
      if (relu && sum < 0.0) sum = 0.0;
      out[o] = sum;
    }
    tensors[outputIndex] = out;
  }
}

class _SoftmaxOp implements _TfliteOp {
  const _SoftmaxOp({
    required this.inputIndex,
    required this.outputIndex,
    required this.beta,
  });

  final int inputIndex;
  final int outputIndex;
  final double beta;

  @override
  void execute(List<Float32List?> tensors, List<List<int>> shapes) {
    final x = tensors[inputIndex]!;
    var maxVal = x[0];
    for (var i = 1; i < x.length; i++) {
      if (x[i] > maxVal) maxVal = x[i];
    }
    final exps = Float64List(x.length);
    var sum = 0.0;
    for (var i = 0; i < x.length; i++) {
      final v = math.exp((x[i] - maxVal) * beta);
      exps[i] = v;
      sum += v;
    }
    final out = Float32List(x.length);
    for (var i = 0; i < x.length; i++) {
      out[i] = exps[i] / sum;
    }
    tensors[outputIndex] = out;
  }
}

class _FlatBufferReader {
  _FlatBufferReader(this.bytes) : bd = ByteData.sublistView(bytes);

  final Uint8List bytes;
  final ByteData bd;

  int u8(int offset) => bd.getUint8(offset);
  int i8(int offset) => bd.getInt8(offset);
  int u16(int offset) => bd.getUint16(offset, Endian.little);
  int u32(int offset) => bd.getUint32(offset, Endian.little);
  int i32(int offset) => bd.getInt32(offset, Endian.little);
  double f32(int offset) => bd.getFloat32(offset, Endian.little);

  int indirect(int offset) => offset + u32(offset);

  int field(int tableOffset, int slot) {
    final vtable = tableOffset - i32(tableOffset);
    final vlen = u16(vtable);
    final slotOffset = 4 + slot * 2;
    if (slotOffset >= vlen) return 0;
    final fieldOffset = u16(vtable + slotOffset);
    return fieldOffset == 0 ? 0 : tableOffset + fieldOffset;
  }

  int vecLen(int vectorFieldOffset) {
    if (vectorFieldOffset == 0) return 0;
    return u32(indirect(vectorFieldOffset));
  }

  int vecTableAt(int vectorFieldOffset, int index) {
    final vecBase = indirect(vectorFieldOffset);
    return indirect(vecBase + 4 + index * 4);
  }

  List<int> i32Vector(int vectorFieldOffset) {
    if (vectorFieldOffset == 0) return const [];
    final vecBase = indirect(vectorFieldOffset);
    final n = u32(vecBase);
    return <int>[for (var i = 0; i < n; i++) i32(vecBase + 4 + i * 4)];
  }

  Uint8List byteVector(int vectorFieldOffset) {
    final vecBase = indirect(vectorFieldOffset);
    final n = u32(vecBase);
    return Uint8List.sublistView(bytes, vecBase + 4, vecBase + 4 + n);
  }
}
