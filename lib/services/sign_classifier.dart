import 'dart:collection';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

/// Coordinates are upright and measured in units of image height on BOTH axes.
typedef SignPoint = ({double x, double y});
typedef SignFrame = List<List<SignPoint>>;

/// Additional measurements in image-height units. Pose order is nose,
/// shoulders, then elbows. Depth slots match the corresponding hand lists.
class BodySignFrame extends ListBase<List<SignPoint>> {
  BodySignFrame(this.hands, this.pose, this.depth);
  final SignFrame hands;
  final List<SignPoint> pose;
  final List<List<double>> depth;
  @override
  int get length => hands.length;
  @override
  set length(int value) => throw UnsupportedError('Immutable frame');
  @override
  List<SignPoint> operator [](int index) => hands[index];
  @override
  void operator []=(int index, List<SignPoint> value) =>
      throw UnsupportedError('Immutable frame');
}

class SignPrediction {
  const SignPrediction(this.label, this.score, [this.minimumScore]);
  final String label;
  final double score;
  final double? minimumScore;
}

class SignClassifier {
  SignClassifier._(
    this.version,
    this.steps,
    this.hidden,
    this.features,
    this.labels,
    this.labelIds,
    this.threshold,
    this.motionFloors,
    this.w,
    this.u,
    this.b,
    this.out,
    this.bias, [
    this.greetings,
    this.fallback,
  ])  : _hScratch = Float64List(hidden),
        _meanScratch = Float64List(features),
        _inputScratch = Float64List(3 * hidden),
        _recurrentScratch = Float64List(3 * hidden),
        _combinedScratch = Float64List(hidden + features),
        _directScratch = Float64List(labels.length),
        _mirrorScratch = Float64List(labels.length);

  static const int _v3StaticFeatures = 142;
  static const List<int> _v3VelocitySources = [
    84, 85, 86, 87,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    8, 9, 16, 17, 24, 25, 32, 33, 40, 41,
    50, 51, 58, 59, 66, 67, 74, 75, 82, 83,
  ];
  static final List<int> _v3MirrorFlipIndices = _buildV3MirrorFlipIndices();

  static const int _v4StaticFeatures = 213;
  static const List<int> _v4VelocitySources = [
    84, 85, 86, 87,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    8, 9, 16, 17, 24, 25, 32, 33, 40, 41,
    50, 51, 58, 59, 66, 67, 74, 75, 82, 83,
    142, 143, 144, 145, 146, 147,
    148, 149, 150, 151, 152, 153, 154, 155,
    166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
    188, 189, 190, 191, 192, 193,
    210, 211, 212,
  ];
  static final List<int> _v4MirrorFlipIndices = _buildV4MirrorFlipIndices();
  static const List<(int, int, int, int)> _curlChains = [
    (1, 2, 2, 4),
    (5, 6, 6, 8),
    (9, 10, 10, 12),
    (13, 14, 14, 16),
    (17, 18, 18, 20),
  ];
  static const List<(int, int, int, int)> _mcpChains = [
    (0, 1, 1, 2),
    (0, 5, 5, 6),
    (0, 9, 9, 10),
    (0, 13, 13, 14),
    (0, 17, 17, 18),
  ];

  static List<int> _buildV3MirrorFlipIndices() {
    final staticFlip = <int>{
      for (var i = 0; i < 88; i += 2) i,
      for (var i = 91; i < 111; i += 2) i,
      140,
    };
    final all = <int>{...staticFlip};
    for (var k = 0; k < _v3VelocitySources.length; k++) {
      if (staticFlip.contains(_v3VelocitySources[k])) {
        all.add(_v3StaticFeatures + k);
      }
    }
    return all.toList()..sort();
  }

  static List<int> _buildV4MirrorFlipIndices() {
    final staticFlip = <int>{
      for (var i = 0; i < 88; i += 2) i,
      for (var i = 91; i < 111; i += 2) i,
      140,
      166, 168, 170, 171, 173, 175,
      176, 179, 182, 185,
      188, 192,
    };
    final all = <int>{...staticFlip};
    for (var k = 0; k < _v4VelocitySources.length; k++) {
      if (staticFlip.contains(_v4VelocitySources[k])) {
        all.add(_v4StaticFeatures + k);
      }
    }
    return all.toList()..sort();
  }

  final int version, steps, hidden, features;
  final SignClassifier? greetings;
  final SignClassifier? fallback;
  final List<String> labels;
  final List<int>? labelIds;
  final double threshold;
  final Map<String, double> motionFloors;
  final Float64List w, u, b, out, bias;
  final Float64List _hScratch;
  final Float64List _meanScratch;
  final Float64List _inputScratch;
  final Float64List _recurrentScratch;
  final Float64List _combinedScratch;
  final Float64List _directScratch;
  final Float64List _mirrorScratch;

  factory SignClassifier.decode(String source) {
    final json = jsonDecode(source) as Map<String, dynamic>;
    final labels = List<String>.from(json['labels'] as List);
    final rawIds = json['labelIds'];
    final labelIds = rawIds == null ? null : List<int>.from(rawIds as List);
    final version = json['version'] as int? ?? 1;
    final features = json['features'] as int;
    final steps = json['steps'] as int;
    final hidden = json['hidden'] as int;
    final validV1 =
        version == 1 && features == 91 && steps == 12 && hidden == 64;
    final validV2 =
        version == 2 && features == 114 && steps == 12 && hidden == 64;
    final validV3 =
        version == 3 && features == 206 && steps == 16 && hidden == 128;
    final validV4 =
        version == 4 && features == 322 && steps == 16 && hidden == 128;
    if (!(validV1 || validV2 || validV3 || validV4) || labels.isEmpty) {
      throw const FormatException('Unrecognized sign model');
    }
    Float64List weights(String key, int length) {
      final values = Float64List.fromList(List<double>.from(json[key] as List));
      if (values.length != length) throw FormatException('Invalid $key');
      return values;
    }

    return SignClassifier._(
        version,
        steps,
        hidden,
        features,
        labels,
        labelIds,
        (json['threshold'] as num).toDouble(),
        {
          for (final e in ((json['motionFloors'] as Map?) ?? {}).entries)
            e.key as String: (e.value as num).toDouble()
        },
        weights('w', features * 3 * hidden),
        weights('u', hidden * 3 * hidden),
        weights('b', 3 * hidden),
        weights('out', (hidden + features) * labels.length),
        weights('bias', labels.length));
  }

  SignClassifier withGreetings(SignClassifier specialist) {
    if (features != specialist.features || specialist.greetings != null) {
      throw ArgumentError('Incompatible greeting model');
    }
    return SignClassifier._(
      version,
      steps,
      hidden,
      features,
      labels,
      labelIds,
      threshold,
      motionFloors,
      w,
      u,
      b,
      out,
      bias,
      specialist,
      fallback,
    );
  }

  SignClassifier withFallback(SignClassifier secondary) {
    return SignClassifier._(
      version,
      steps,
      hidden,
      features,
      labels,
      labelIds,
      threshold,
      motionFloors,
      w,
      u,
      b,
      out,
      bias,
      greetings,
      secondary,
    );
  }

  SignPrediction? predict(List<SignFrame> frames) {
    final (evaluated, primary) = switch (version) {
      >= 4 => _predictV4(frames),
      3 => (false, _predictV3(frames)),
      _ => (false, _predictLegacy(frames)),
    };
    final secondary = fallback;
    if (secondary == null) return primary;
    final superset = secondary.labels.every(labels.contains);
    // When unified v4 already evaluated valid body+hand frames and covers
    // every fallback label, never bypass motionFloors/suppression via legacy.
    if (evaluated && superset) return primary;
    if (primary != null && (accepts(primary) || superset)) {
      return primary;
    }
    final backup = secondary.predict(frames);
    if (backup != null && secondary.accepts(backup)) return backup;
    return primary ?? backup;
  }

  /// Legacy v1 (91d hands-only) and v2 (114d upper-body) feature extraction.
  SignPrediction? _predictLegacy(List<SignFrame> frames) {
    final visible = <SignFrame>[];
    final bodyFrames = <BodySignFrame>[];
    final scales = <double>[];
    for (final frame in frames) {
      final hands = frame
          .where((h) =>
              h.length == 21 &&
              h.every((p) => p.x.isFinite && p.y.isFinite) &&
              _palm(h) > 0.0001)
          .toList()
        ..sort((a, b) => a.first.y.compareTo(b.first.y));
      if (hands.isEmpty) continue;
      if (features > 91) {
        if (frame is! BodySignFrame) return null;
        bodyFrames.add(frame);
      }
      visible.add(hands);
      scales.add(_palm(hands.first));
    }
    if (visible.length < 3) return null;
    scales.sort();
    final mid = scales.length ~/ 2;
    final scale = (scales[mid] + scales[(scales.length - 1) ~/ 2]) / 2;
    final origin = visible.first.first.first;
    final rows = <Float64List>[];
    for (var i = 0; i < visible.length; i++) {
      final hands = visible[i];
      final row = Float64List(features);
      for (var h = 0; h < math.min(2, hands.length); h++) {
        final hand = hands[h], wrist = hand.first;
        final palm = _palm(hand);
        for (var j = 0; j < 21; j++) {
          row[h * 42 + j * 2] = (hand[j].x - wrist.x) / palm;
          row[h * 42 + j * 2 + 1] = (hand[j].y - wrist.y) / palm;
        }
      }
      final wrist = hands.first.first;
      row[84] = (wrist.x - origin.x) / scale;
      row[85] = (wrist.y - origin.y) / scale;
      if (hands.length > 1) {
        row[86] = (hands[1].first.x - wrist.x) / scale;
        row[87] = (hands[1].first.y - wrist.y) / scale;
        row[88] = 1;
      }
      row[89] = math.sqrt(row[84] * row[84] + row[85] * row[85]);
      row[90] = math.sqrt(row[86] * row[86] + row[87] * row[87]);
      if (features > 91) _addBody(row, bodyFrames[i], hands);
      for (var j = 0; j < features; j++) {
        row[j] = row[j].clamp(-5.0, 5.0);
      }
      rows.add(row);
    }
    if (features > 91 && rows.every((r) => r[103] == 0)) return null;
    final sequence = _resampleRows(rows, features);
    final reflected = [for (final row in sequence) Float64List.fromList(row)];
    for (final row in reflected) {
      for (var i = 0; i < 88; i += 2) {
        row[i] = -row[i];
      }
      for (var i = 91; features > 91 && i < 103; i += 2) {
        row[i] = -row[i];
      }
    }
    final original = _score(sequence, reflected);
    final specialist = greetings;
    if (specialist == null) return original;
    if (original != null &&
        original.score >= threshold &&
        !specialist.labels.contains(original.label) &&
        original.label != 'WELCOME') {
      return original;
    }
    final greeting = specialist._score(sequence, reflected);
    if (greeting != null && greeting.score >= specialist.threshold) {
      return greeting;
    }
    return original;
  }

  /// V3 (206d) feature extraction with stabilized 2-hand slots, carried-forward
  /// upper-body reference, head/face fingertip offsets, openness/pinch ratios,
  /// and explicit temporal velocity channels.
  SignPrediction? _predictV3(List<SignFrame> frames) {
    final visibleSlots = <List<_HandSlot>>[];
    final rawPoses = <List<SignPoint>>[];
    for (final frame in frames) {
      if (frame is! BodySignFrame) return null;
      final candidates = <_HandSlot>[];
      for (var origIdx = 0; origIdx < frame.length; origIdx++) {
        final hand = frame[origIdx];
        if (hand.length != 21 ||
            !hand.every((p) => p.x.isFinite && p.y.isFinite) ||
            _palm(hand) <= 0.0001) {
          continue;
        }
        final depth = origIdx < frame.depth.length
            ? frame.depth[origIdx]
            : const <double>[];
        candidates.add(_HandSlot(hand, depth));
      }
      if (candidates.isEmpty) continue;
      candidates.sort((a, b) => a.hand.first.y.compareTo(b.hand.first.y));
      visibleSlots.add(candidates.take(2).toList());
      rawPoses.add(frame.pose);
    }
    if (visibleSlots.length < 3) return null;

    // Stabilize two-hand slot assignment across close vertical crossings.
    for (var i = 1; i < visibleSlots.length; i++) {
      final curr = visibleSlots[i];
      final prev = visibleSlots[i - 1];
      if (curr.length == 2 && prev.length == 2) {
        final w0 = curr[0].hand.first, w1 = curr[1].hand.first;
        final pw0 = prev[0].hand.first, pw1 = prev[1].hand.first;
        final palm = _palm(curr[0].hand);
        if ((w0.y - w1.y).abs() < 0.45 * math.max(palm, 0.02)) {
          final dKeep = _sqDist(w0, pw0) + _sqDist(w1, pw1);
          final dSwap = _sqDist(w1, pw0) + _sqDist(w0, pw1);
          if (dSwap + 1e-6 < dKeep) {
            visibleSlots[i] = [curr[1], curr[0]];
          }
        }
      }
    }

    final resolvedPoses = _resolvePoses(rawPoses);
    if (resolvedPoses == null) return null;

    final scales = [for (final s in visibleSlots) _palm(s[0].hand)]..sort();
    final mid = scales.length ~/ 2;
    final primaryScale = math.max(
        (scales[mid] + scales[(scales.length - 1) ~/ 2]) / 2, 0.0001);
    final origin = visibleSlots.first.first.hand.first;

    final rows = <Float64List>[];
    for (var i = 0; i < visibleSlots.length; i++) {
      final slots = visibleSlots[i];
      final pose = resolvedPoses[i];
      final row = Float64List(features);
      final cx = (pose[1].x + pose[2].x) / 2;
      final cy = (pose[1].y + pose[2].y) / 2;
      final shoulderWidth = math.max(_dist(pose[1], pose[2]), 0.025);

      for (var h = 0; h < slots.length; h++) {
        final hand = slots[h].hand, depth = slots[h].depth;
        final wrist = hand.first;
        final palm = math.max(_palm(hand), 0.0001);

        for (var j = 0; j < 21; j++) {
          row[h * 42 + j * 2] = (hand[j].x - wrist.x) / palm;
          row[h * 42 + j * 2 + 1] = (hand[j].y - wrist.y) / palm;
        }

        final elbow =
            _dist(wrist, pose[3]) <= _dist(wrist, pose[4]) ? pose[3] : pose[4];
        final bodyBase = 91 + h * 10;
        row[bodyBase] = (wrist.x - cx) / shoulderWidth;
        row[bodyBase + 1] = (wrist.y - cy) / shoulderWidth;
        row[bodyBase + 2] = (elbow.x - cx) / shoulderWidth;
        row[bodyBase + 3] = (elbow.y - cy) / shoulderWidth;
        row[bodyBase + 4] = (wrist.x - pose[0].x) / shoulderWidth;
        row[bodyBase + 5] = (wrist.y - pose[0].y) / shoulderWidth;
        row[bodyBase + 6] = (hand[8].x - pose[0].x) / shoulderWidth;
        row[bodyBase + 7] = (hand[8].y - pose[0].y) / shoulderWidth;
        row[bodyBase + 8] = (hand[4].x - pose[0].x) / shoulderWidth;
        row[bodyBase + 9] = (hand[4].y - pose[0].y) / shoulderWidth;

        if (depth.length >= 21) {
          for (var j = 0; j < 5; j++) {
            row[112 + h * 5 + j] = (depth[4 + j * 4] - depth[0]) / palm;
          }
        }
        for (var j = 0; j < 5; j++) {
          row[122 + h * 5 + j] = _dist(hand[4 + j * 4], wrist) / palm;
        }
        for (var j = 0; j < 4; j++) {
          row[132 + h * 4 + j] = _dist(hand[8 + j * 4], hand[4]) / palm;
        }
      }

      final w0 = slots[0].hand.first;
      row[84] = (w0.x - origin.x) / primaryScale;
      row[85] = (w0.y - origin.y) / primaryScale;
      if (slots.length > 1) {
        final w1 = slots[1].hand.first;
        row[86] = (w1.x - w0.x) / primaryScale;
        row[87] = (w1.y - w0.y) / primaryScale;
        row[88] = 1.0;
        row[140] = (slots[1].hand[8].x - slots[0].hand[8].x) / primaryScale;
        row[141] = (slots[1].hand[8].y - slots[0].hand[8].y) / primaryScale;
      }
      row[89] = math.sqrt(row[84] * row[84] + row[85] * row[85]);
      row[90] = math.sqrt(row[86] * row[86] + row[87] * row[87]);
      row[111] = 1.0;

      for (var j = 0; j < _v3StaticFeatures; j++) {
        row[j] = row[j].clamp(-5.0, 5.0);
      }
      rows.add(row);
    }

    final sequence = _resampleRows(rows, _v3StaticFeatures);
    for (var s = 1; s < steps; s++) {
      final curr = sequence[s], prev = sequence[s - 1];
      for (var k = 0; k < _v3VelocitySources.length; k++) {
        final src = _v3VelocitySources[k];
        curr[_v3StaticFeatures + k] =
            ((curr[src] - prev[src]) * 3.0).clamp(-5.0, 5.0);
      }
    }

    final reflected = [for (final row in sequence) Float64List.fromList(row)];
    for (final row in reflected) {
      for (final flipIdx in _v3MirrorFlipIndices) {
        row[flipIdx] = -row[flipIdx];
      }
    }
    return _score(sequence, reflected);
  }

  /// V4 (322d) feature extraction with main-island filtering, duplicate-hand
  /// suppression, wider crossing hysteresis, palm/arm orientation, finger curls,
  /// thumb-tip spreads, MCP flexion, head-height normalization, and 2-hand coordination.
  (bool, SignPrediction?) _predictV4(List<SignFrame> frames) {
    final rawEntries = <_V4FrameEntry>[];
    for (var frameIdx = 0; frameIdx < frames.length; frameIdx++) {
      final frame = frames[frameIdx];
      if (frame is! BodySignFrame) return (false, null);
      final candidates = <_HandSlot>[];
      for (var origIdx = 0; origIdx < frame.length; origIdx++) {
        final hand = frame[origIdx];
        if (hand.length != 21 ||
            !hand.every((p) => p.x.isFinite && p.y.isFinite) ||
            !hand.any((p) => p.x != 0 || p.y != 0) ||
            _palm(hand) <= 0.0001) {
          continue;
        }
        final depth = origIdx < frame.depth.length
            ? frame.depth[origIdx]
            : const <double>[];
        candidates.add(_HandSlot(hand, depth, origIdx));
      }
      if (candidates.isEmpty) continue;
      candidates.sort((a, b) {
        final c = a.hand.first.y.compareTo(b.hand.first.y);
        return c != 0 ? c : a.origSlot.compareTo(b.origSlot);
      });
      final topSlots = candidates.take(2).toList();
      if (topSlots.length == 2) {
        final h0 = topSlots[0].hand, h1 = topSlots[1].hand;
        final palm = math.max(math.max(_palm(h0), _palm(h1)), 1e-4);
        var sumD = 0.0;
        for (var j = 0; j < 21; j++) {
          sumD += _dist(h0[j], h1[j]);
        }
        if (sumD / 21.0 < 0.45 * palm) {
          topSlots.removeAt(1);
        }
      }
      rawEntries.add(_V4FrameEntry(frameIdx, topSlots, frame.pose));
    }

    final keptEntries = _filterMainIslands(rawEntries, maxGap: 10);
    if (keptEntries.length < 3) return (false, null);

    final visibleSlots = [for (final e in keptEntries) e.slots];
    final rawPoses = [for (final e in keptEntries) e.pose];
    final resolvedPoses = _resolvePoses(rawPoses);
    if (resolvedPoses == null) return (false, null);

    for (var i = 1; i < visibleSlots.length; i++) {
      final curr = visibleSlots[i];
      final prev = visibleSlots[i - 1];
      if (curr.length == 2 && prev.length == 2) {
        final w0 = curr[0].hand.first, w1 = curr[1].hand.first;
        final pw0 = prev[0].hand.first, pw1 = prev[1].hand.first;
        final palm = math.max(_palm(curr[0].hand), 0.02);
        if ((w0.y - w1.y).abs() < 1.25 * palm) {
          final dKeep = _sqDist(w0, pw0) + _sqDist(w1, pw1);
          final dSwap = _sqDist(w1, pw0) + _sqDist(w0, pw1);
          if (dSwap + 1e-6 < dKeep) {
            visibleSlots[i] = [curr[1], curr[0]];
          }
        }
      }
    }

    final scales = [for (final s in visibleSlots) _palm(s[0].hand)]..sort();
    final mid = scales.length ~/ 2;
    final primaryScale = math.max(
        (scales[mid] + scales[(scales.length - 1) ~/ 2]) / 2, 1e-4);
    final anchorW0 = visibleSlots.first[0].hand.first;
    var sumW0x = 0.0, sumW0y = 0.0;
    for (final slots in visibleSlots) {
      sumW0x += slots[0].hand.first.x;
      sumW0y += slots[0].hand.first.y;
    }
    final meanW0 = (
      x: sumW0x / visibleSlots.length,
      y: sumW0y / visibleSlots.length,
    );

    final rows = <Float64List>[];
    for (var i = 0; i < visibleSlots.length; i++) {
      final slots = visibleSlots[i];
      final pose = resolvedPoses[i];
      final row = Float64List(features);
      final cx = (pose[1].x + pose[2].x) / 2;
      final cy = (pose[1].y + pose[2].y) / 2;
      final shoulderWidth = math.max(_dist(pose[1], pose[2]), 0.025);
      final headH = math.max(
        _dist((x: cx, y: cy), pose[0]),
        0.25 * shoulderWidth,
      );

      for (var h = 0; h < slots.length; h++) {
        final hand = slots[h].hand, depth = slots[h].depth;
        final wrist = hand.first;
        final palm = math.max(_palm(hand), 1e-4);
        final normPts = List<SignPoint>.generate(
          21,
          (j) {
            final nx = (hand[j].x - wrist.x) / palm;
            final ny = (hand[j].y - wrist.y) / palm;
            row[h * 42 + j * 2] = nx;
            row[h * 42 + j * 2 + 1] = ny;
            return (x: nx, y: ny);
          },
          growable: false,
        );

        final nearLeft = _dist(wrist, pose[3]) <= _dist(wrist, pose[4]);
        final elbow = nearLeft ? pose[3] : pose[4];
        final shoulder = nearLeft ? pose[1] : pose[2];
        if (h == 0) {
          final oppElbow = nearLeft ? pose[4] : pose[3];
          row[192] = (wrist.x - oppElbow.x) / shoulderWidth;
          row[193] = (wrist.y - oppElbow.y) / shoulderWidth;
        }

        final bodyBase = 91 + h * 10;
        row[bodyBase] = (wrist.x - cx) / shoulderWidth;
        row[bodyBase + 1] = (wrist.y - cy) / shoulderWidth;
        row[bodyBase + 2] = (elbow.x - cx) / shoulderWidth;
        row[bodyBase + 3] = (elbow.y - cy) / shoulderWidth;
        row[bodyBase + 4] = (wrist.x - pose[0].x) / shoulderWidth;
        row[bodyBase + 5] = (wrist.y - pose[0].y) / shoulderWidth;
        row[bodyBase + 6] = (hand[8].x - pose[0].x) / shoulderWidth;
        row[bodyBase + 7] = (hand[8].y - pose[0].y) / shoulderWidth;
        row[bodyBase + 8] = (hand[4].x - pose[0].x) / shoulderWidth;
        row[bodyBase + 9] = (hand[4].y - pose[0].y) / shoulderWidth;

        if (depth.length >= 21) {
          for (var j = 0; j < 5; j++) {
            row[112 + h * 5 + j] = (depth[4 + j * 4] - depth[0]) / palm;
          }
        }
        for (var j = 0; j < 5; j++) {
          row[122 + h * 5 + j] = math.sqrt(
            normPts[4 + j * 4].x * normPts[4 + j * 4].x +
                normPts[4 + j * 4].y * normPts[4 + j * 4].y,
          );
        }
        for (var j = 0; j < 4; j++) {
          row[132 + h * 4 + j] = _dist(normPts[8 + j * 4], normPts[4]);
        }

        final spreadBase = 142 + h * 3;
        row[spreadBase] = _dist(normPts[8], normPts[12]);
        row[spreadBase + 1] = _dist(normPts[12], normPts[16]);
        row[spreadBase + 2] = _dist(normPts[16], normPts[20]);

        final thumbPipBase = 148 + h * 4;
        row[thumbPipBase] = _dist(normPts[4], normPts[5]);
        row[thumbPipBase + 1] = _dist(normPts[4], normPts[6]);
        row[thumbPipBase + 2] = _dist(normPts[4], normPts[10]);
        row[thumbPipBase + 3] = _dist(normPts[4], normPts[14]);

        final thumbTipsBase = 194 + h * 3;
        row[thumbTipsBase] = _dist(normPts[4], normPts[12]);
        row[thumbTipsBase + 1] = _dist(normPts[4], normPts[16]);
        row[thumbTipsBase + 2] = _dist(normPts[4], normPts[20]);

        _writeSegmentCosines(normPts, _curlChains, row, 156 + h * 5);
        _writeSegmentCosines(normPts, _mcpChains, row, 200 + h * 5);

        final uPx = normPts[9].x;
        final uPy = normPts[9].y;
        final uPNorm = math.max(math.sqrt(uPx * uPx + uPy * uPy), 1e-6);
        final ux = uPx / uPNorm, uy = uPy / uPNorm;
        final vKx = normPts[17].x - normPts[5].x;
        final vKy = normPts[17].y - normPts[5].y;
        final vKNorm = math.max(math.sqrt(vKx * vKx + vKy * vKy), 1e-6);
        final vx = vKx / vKNorm, vy = vKy / vKNorm;
        final palmBase = 166 + h * 5;
        row[palmBase] = ux;
        row[palmBase + 1] = uy;
        row[palmBase + 2] = vx;
        row[palmBase + 3] = vy;
        row[palmBase + 4] = ux * vy - uy * vx;

        final fax = wrist.x - elbow.x;
        final fay = wrist.y - elbow.y;
        final faLenRaw = math.sqrt(fax * fax + fay * fay);
        final faNorm = math.max(faLenRaw, 0.02);
        final uax = elbow.x - shoulder.x;
        final uay = elbow.y - shoulder.y;
        final uaNorm = math.max(math.sqrt(uax * uax + uay * uay), 0.02);
        final armBase = 176 + h * 6;
        row[armBase] = fax / faNorm;
        row[armBase + 1] = fay / faNorm;
        row[armBase + 2] = faLenRaw / shoulderWidth;
        row[armBase + 3] = uax / uaNorm;
        row[armBase + 4] = uay / uaNorm;
        row[armBase + 5] = (wrist.y - pose[0].y) / headH;
      }

      final w0 = slots[0].hand.first;
      row[84] = (w0.x - anchorW0.x) / primaryScale;
      row[85] = (w0.y - anchorW0.y) / primaryScale;
      row[89] = math.sqrt(row[84] * row[84] + row[85] * row[85]);
      row[111] = 1.0;
      row[188] = (w0.x - meanW0.x) / shoulderWidth;
      row[189] = (w0.y - meanW0.y) / shoulderWidth;
      row[190] = (slots[0].hand[8].y - pose[0].y) / headH;
      row[191] = (slots[0].hand[4].y - pose[0].y) / headH;
      if (slots.length > 1) {
        final w1 = slots[1].hand.first;
        row[86] = (w1.x - w0.x) / primaryScale;
        row[87] = (w1.y - w0.y) / primaryScale;
        row[88] = 1.0;
        row[90] = math.sqrt(row[86] * row[86] + row[87] * row[87]);
        row[140] = (slots[1].hand[8].x - slots[0].hand[8].x) / primaryScale;
        row[141] = (slots[1].hand[8].y - slots[0].hand[8].y) / primaryScale;
        row[210] = row[166] * row[171] + row[167] * row[172];
        row[211] = _dist(slots[0].hand[9], slots[1].hand[9]) / primaryScale;
        row[212] = _dist(slots[0].hand[8], slots[1].hand[0]) / primaryScale;
      }

      for (var j = 0; j < _v4StaticFeatures; j++) {
        row[j] = row[j].clamp(-5.0, 5.0);
      }
      rows.add(row);
    }

    final sequence = _resampleRows(rows, _v4StaticFeatures);
    for (var s = 1; s < steps; s++) {
      final curr = sequence[s], prev = sequence[s - 1];
      for (var k = 0; k < _v4VelocitySources.length; k++) {
        final src = _v4VelocitySources[k];
        curr[_v4StaticFeatures + k] =
            ((curr[src] - prev[src]) * 3.0).clamp(-5.0, 5.0);
      }
    }

    final reflected = [for (final row in sequence) Float64List.fromList(row)];
    for (final row in reflected) {
      for (final flipIdx in _v4MirrorFlipIndices) {
        row[flipIdx] = -row[flipIdx];
      }
    }
    return (true, _score(sequence, reflected));
  }

  static void _writeSegmentCosines(
    List<SignPoint> pts,
    List<(int, int, int, int)> chains,
    Float64List row,
    int base,
  ) {
    for (var j = 0; j < chains.length; j++) {
      final (a, b, c, d) = chains[j];
      final ux = pts[b].x - pts[a].x, uy = pts[b].y - pts[a].y;
      final vx = pts[d].x - pts[c].x, vy = pts[d].y - pts[c].y;
      final denom = math.max(
        math.sqrt(ux * ux + uy * uy) * math.sqrt(vx * vx + vy * vy),
        1e-6,
      );
      row[base + j] = (ux * vx + uy * vy) / denom;
    }
  }

  static int _roundHalfToEven(double value) {
    final floor = value.floor();
    final frac = value - floor;
    if (frac > 0.5) return floor + 1;
    if (frac < 0.5) return floor;
    return floor.isEven ? floor : floor + 1;
  }

  static List<_V4FrameEntry> _filterMainIslands(
    List<_V4FrameEntry> entries, {
    int maxGap = 10,
  }) {
    if (entries.isEmpty) return entries;
    final islands = <List<_V4FrameEntry>>[];
    var start = 0;
    for (var i = 1; i < entries.length; i++) {
      if (entries[i].frameIdx - entries[i - 1].frameIdx >= maxGap) {
        islands.add(entries.sublist(start, i));
        start = i;
      }
    }
    islands.add(entries.sublist(start));
    if (islands.length <= 1) return entries;

    var best = 0;
    var bestIdx = 0;
    for (var i = 0; i < islands.length; i++) {
      if (islands[i].length > best) {
        best = islands[i].length;
        bestIdx = i;
      }
    }
    final minLen = math.max(8, _roundHalfToEven(0.35 * best));
    final kept = <_V4FrameEntry>[];
    for (final isl in islands) {
      if (isl.length >= minLen) kept.addAll(isl);
    }
    return kept.isEmpty ? islands[bestIdx] : kept;
  }

  List<List<SignPoint>>? _resolvePoses(List<List<SignPoint>> rawPoses) {
    bool isValid(List<SignPoint> p) =>
        p.length == 5 &&
        p.every((pt) => pt.x.isFinite && pt.y.isFinite && (pt.x != 0 || pt.y != 0)) &&
        _dist(p[1], p[2]) > 0.025;

    final out = List<List<SignPoint>?>.filled(rawPoses.length, null);
    List<SignPoint>? lastValid;
    int? firstValidIdx;
    for (var i = 0; i < rawPoses.length; i++) {
      if (isValid(rawPoses[i])) {
        lastValid = rawPoses[i];
        firstValidIdx ??= i;
        out[i] = lastValid;
      } else if (lastValid != null) {
        out[i] = lastValid;
      }
    }
    if (firstValidIdx == null) return null;
    for (var i = 0; i < firstValidIdx; i++) {
      out[i] = out[firstValidIdx];
    }
    return [for (final p in out) p!];
  }

  List<Float64List> _resampleRows(List<Float64List> rows, int activeCols) {
    return List.generate(steps, (i) {
      final t = i * (rows.length - 1) / (steps - 1);
      final loIdx = t.floor(), hiIdx = t.ceil();
      final lo = rows[loIdx], hi = rows[hiIdx];
      final frac = t - loIdx;
      final outRow = Float64List(features);
      for (var j = 0; j < activeCols; j++) {
        outRow[j] = lo[j] + (hi[j] - lo[j]) * frac;
      }
      return outRow;
    });
  }

  bool accepts(SignPrediction result) {
    return result.score >= (result.minimumScore ?? threshold);
  }

  SignPrediction? _score(
      List<Float64List> sequence, List<Float64List> reflected) {
    final direct = _logitsInto(sequence, _directScratch);
    final mirror = _logitsInto(reflected, _mirrorScratch);
    var best = 0;
    final labelCount = labels.length;
    for (var i = 0; i < labelCount; i++) {
      direct[i] = (direct[i] + mirror[i]) / 2;
      if (direct[i] > direct[best]) best = i;
    }
    final maxLogit = direct[best];
    var sum = 0.0;
    for (var i = 0; i < labelCount; i++) {
      sum += math.exp(direct[i] - maxLogit);
    }
    if (labels[best] == '__OTHER__') return null;
    final floor = motionFloors[labels[best]];
    if (floor != null && _motion(sequence) < floor) return null;
    return SignPrediction(labels[best], 1 / sum, threshold);
  }

  static double _span(List<double> values) {
    values.sort();
    double quantile(double q) {
      final pos = (values.length - 1) * q;
      return values[pos.floor()] +
          (values[pos.ceil()] - values[pos.floor()]) * (pos - pos.floor());
    }

    return quantile(.8) - quantile(.2);
  }

  double _motion(List<Float64List> sequence) {
    final minRows = version >= 3 ? 8 : 6;
    final bodyValidIdx = version >= 3 ? 111 : 103;
    final bodyStride = version >= 3 ? 10 : 6;
    final depthBase = version >= 3 ? 112 : 104;

    double slotMotion(List<Float64List> rows, int slot) {
      var tips = 0.0;
      for (var j = 0; j < 5; j++) {
        final index = slot * 42 + (4 + 4 * j) * 2;
        final x = _span([for (final r in rows) r[index]]);
        final y = _span([for (final r in rows) r[index + 1]]);
        final z = _span([for (final r in rows) r[depthBase + slot * 5 + j]]);
        tips += math.sqrt(x * x + y * y + z * z);
      }
      var peak = tips / 5;
      final body = rows.where((r) => r[bodyValidIdx] > .95).toList();
      if (body.length >= minRows) {
        final x = _span([for (final r in body) r[91 + slot * bodyStride]]);
        final y = _span([for (final r in body) r[92 + slot * bodyStride]]);
        peak = math.max(peak, math.sqrt(x * x + y * y));
      }
      return peak;
    }

    double best = 0;
    for (final two in [false, true]) {
      final rows =
          sequence.where((r) => two ? r[88] > .95 : r[88] < .05).toList();
      if (rows.length < minRows) continue;
      for (var slot = 0; slot < (two ? 2 : 1); slot++) {
        best = math.max(best, slotMotion(rows, slot));
      }
    }
    if (version >= 3 && best == 0 && sequence.length >= minRows) {
      best = slotMotion(sequence, 0);
    }
    return best;
  }

  static void _addBody(Float64List row, BodySignFrame frame, SignFrame sorted) {
    final pose = frame.pose;
    final valid = pose.length == 5 &&
        pose.every(
            (p) => p.x.isFinite && p.y.isFinite && (p.x != 0 || p.y != 0));
    final width = valid ? _dist(pose[1], pose[2]) : 0.0;
    for (var slot = 0; slot < sorted.length && slot < 2; slot++) {
      final hand = sorted[slot], wrist = hand.first;
      final depths = frame.depth[frame.hands.indexOf(hand)];
      final palm = _palm(hand);
      for (var j = 0; j < 5; j++) {
        row[104 + slot * 5 + j] = (depths[4 + j * 4] - depths[0]) / palm;
      }
      if (width <= .025) continue;
      row[103] = 1;
      final cx = (pose[1].x + pose[2].x) / 2, cy = (pose[1].y + pose[2].y) / 2;
      final elbow =
          _dist(wrist, pose[3]) <= _dist(wrist, pose[4]) ? pose[3] : pose[4];
      final values = [
        wrist.x - cx,
        wrist.y - cy,
        elbow.x - cx,
        elbow.y - cy,
        wrist.x - pose[0].x,
        wrist.y - pose[0].y
      ];
      for (var j = 0; j < values.length; j++) {
        row[91 + slot * 6 + j] = values[j] / width;
      }
    }
  }

  static double _dist(SignPoint a, SignPoint b) {
    final dx = a.x - b.x, dy = a.y - b.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  static double _sqDist(SignPoint a, SignPoint b) {
    final dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
  }

  static double _palm(List<SignPoint> hand) {
    var sum = 0.0;
    for (final i in const [5, 9, 13, 17]) {
      final dx = hand[i].x - hand[0].x, dy = hand[i].y - hand[0].y;
      sum += math.sqrt(dx * dx + dy * dy);
    }
    return sum / 4;
  }

  static double _sigmoid(double v) => 1 / (1 + math.exp(-v.clamp(-40.0, 40.0)));

  static void _multiplyAdd(
      Float64List x, Float64List matrix, Float64List result) {
    final resultLen = result.length;
    final xLen = x.length;
    for (var i = 0; i < xLen; i++) {
      final v = x[i];
      if (v == 0) continue;
      final start = i * resultLen;
      for (var j = 0; j < resultLen; j++) {
        result[j] += v * matrix[start + j];
      }
    }
  }

  Float64List _logitsInto(List<Float64List> sequence, Float64List result) {
    final h = _hScratch;
    final mean = _meanScratch;
    final input = _inputScratch;
    final recurrent = _recurrentScratch;
    h.fillRange(0, hidden, 0.0);
    mean.fillRange(0, features, 0.0);
    final invSteps = 1.0 / steps;
    final hidden2 = 2 * hidden;
    final hidden3 = 3 * hidden;
    for (final x in sequence) {
      input.setAll(0, b);
      recurrent.fillRange(0, hidden3, 0.0);
      _multiplyAdd(x, w, input);
      _multiplyAdd(h, u, recurrent);
      for (var j = 0; j < hidden; j++) {
        final z = _sigmoid(input[j] + recurrent[j]);
        final r = _sigmoid(input[hidden + j] + recurrent[hidden + j]);
        final n = 2 *
                _sigmoid(2 *
                    (input[hidden2 + j] + r * recurrent[hidden2 + j])) -
            1;
        h[j] = z * h[j] + (1 - z) * n;
      }
      for (var j = 0; j < features; j++) {
        mean[j] += x[j] * invSteps;
      }
    }
    _combinedScratch.setRange(0, hidden, h);
    _combinedScratch.setRange(hidden, hidden + features, mean);
    result.setAll(0, bias);
    _multiplyAdd(_combinedScratch, out, result);
    return result;
  }
}

class _HandSlot {
  const _HandSlot(this.hand, this.depth, [this.origSlot = 0]);
  final List<SignPoint> hand;
  final List<double> depth;
  final int origSlot;
}

class _V4FrameEntry {
  const _V4FrameEntry(this.frameIdx, this.slots, this.pose);
  final int frameIdx;
  final List<_HandSlot> slots;
  final List<SignPoint> pose;
}
