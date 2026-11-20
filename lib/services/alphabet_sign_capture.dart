import 'dart:math' as math;
import 'dart:typed_data';

import 'alphabet_classifier.dart';
import 'landmark_service.dart';
import 'sign_classifier.dart';

/// Stateful rolling-window controller for **Alphabet Mode**.
///
/// Requires a steady handshape held across multiple frames before confirming a
/// letter, clears stale predictions quickly when hands leave the frame,
/// distinguishes low-confidence/unknown handshapes with a retry prompt, and
/// includes trajectory assistance for dynamic letters `J` and `Z`.
class AlphabetSignCapture {
  AlphabetSignCapture(this.classifier);

  static const int windowMs = 1500;
  static const int maxWindowFrames = 8;
  static const int motionWindowMs = 1400;
  static const int minFrames = 3;
  static const int minHoldMs = 130;
  static const int missingWindowResetMs = 240;
  static const int disappearClearMs = 260;
  static const int retryDelayMs = 650;
  static const double maxStepDisplacement = 0.45;
  static const double unlatchStepDisplacement = 0.34;
  static const double unlatchWristShift = 0.32;
  static const double maxNetWristDrift = 0.55;
  static const double minMajorityRatio = 0.60;

  final AlphabetClassifier classifier;
  final List<String> letters = <String>[];

  final List<_AlphabetSample> _window = <_AlphabetSample>[];
  final List<_MotionSample> _motionWindow = <_MotionSample>[];

  SignPrediction? activePrediction;
  bool retry = false;
  bool isDynamicLetterActive = false;

  double _lastStepDelta = 0.0;
  double _lastWristShift = 0.0;
  double _windowMeanStep = 0.0;
  double _windowWristDrift = 0.0;
  double _windowWristPath = 0.0;
  int _windowWristMoveFrames = 0;

  double get lastStepDelta => _lastStepDelta;
  double get lastWristShift => _lastWristShift;
  double get windowMeanStep => _windowMeanStep;
  double get windowWristDrift => _windowWristDrift;
  bool get isPostureSteady =>
      _window.length >= minFrames &&
      _lastStepDelta < 0.22 &&
      _lastWristShift < 0.20 &&
      _windowMeanStep < 0.22 &&
      _windowWristDrift < 0.32 &&
      !(_windowWristMoveFrames >= 4 && _windowWristPath >= 0.50);

  String? _latchedLetter;
  int? _firstMissingMs;
  int? _steadyStartMs;
  int? _lastTimestampMs;
  int? _lastDynamicLetterMs;

  void undo() {
    if (letters.isNotEmpty) letters.removeLast();
  }

  void clear() {
    letters.clear();
    reset();
  }

  void reset() {
    _window.clear();
    _motionWindow.clear();
    activePrediction = null;
    retry = false;
    isDynamicLetterActive = false;
    _lastStepDelta = 0.0;
    _lastWristShift = 0.0;
    _windowMeanStep = 0.0;
    _windowWristDrift = 0.0;
    _windowWristPath = 0.0;
    _windowWristMoveFrames = 0;
    _latchedLetter = null;
    _firstMissingMs = null;
    _steadyStartMs = null;
    _lastTimestampMs = null;
    _lastDynamicLetterMs = null;
  }

  void add(List<List<LandmarkPoint>> hands, int timestampMs) {
    if (_lastTimestampMs != null && timestampMs < _lastTimestampMs!) {
      reset();
    }
    _lastTimestampMs = timestampMs;

    final primary = _selectPrimaryHand(hands);
    if (primary == null) {
      _handleMissingHand(timestampMs);
      return;
    }

    final feats = AlphabetClassifier.extractFeatures(primary.landmarks);
    if (feats == null) {
      _handleMissingHand(timestampMs);
      return;
    }

    _firstMissingMs = null;
    final probs = classifier.predictProbsFromFeatures(feats);

    var bestIdx = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }

    final sample = _AlphabetSample(
      timestampMs: timestampMs,
      wristX: primary.landmarks.first.x,
      wristY: primary.landmarks.first.y,
      palmScale: primary.palmScale,
      indexX: primary.landmarks[8].x,
      indexY: primary.landmarks[8].y,
      pinkyX: primary.landmarks[20].x,
      pinkyY: primary.landmarks[20].y,
      features: feats,
      probs: probs,
      bestLabel: classifier.labels[bestIdx],
      bestScore: probs[bestIdx],
    );

    final motion = _MotionSample(
      timestampMs: timestampMs,
      wristX: sample.wristX,
      wristY: sample.wristY,
      palmScale: sample.palmScale,
      indexX: sample.indexX,
      indexY: sample.indexY,
      pinkyX: sample.pinkyX,
      pinkyY: sample.pinkyY,
      probs: probs,
    );

    final double stepDelta;
    if (_window.isNotEmpty) {
      final prev = _window.last;
      stepDelta = _normalizedDisplacement(prev, sample);
      final meanScale =
          math.max((prev.palmScale + sample.palmScale) * 0.5, 1e-4);
      final wristDx = sample.wristX - prev.wristX;
      final wristDy = sample.wristY - prev.wristY;
      final wristShift =
          math.sqrt(wristDx * wristDx + wristDy * wristDy) / meanScale;
      _lastStepDelta = stepDelta;
      _lastWristShift = wristShift;
      if (_latchedLetter != null &&
          (stepDelta >= unlatchStepDisplacement ||
              wristShift >= unlatchWristShift)) {
        // A crisp positional bounce lets a signer enter repeated identical
        // letters (e.g. L-L in HELLO) without dropping out of frame.
        _latchedLetter = null;
        activePrediction = null;
        isDynamicLetterActive = false;
      }
      if (stepDelta >= maxStepDisplacement ||
          wristShift >= maxStepDisplacement) {
        _steadyStartMs = timestampMs;
        _window.clear();
        _windowMeanStep = 0.0;
        _windowWristDrift = 0.0;
        _windowWristPath = 0.0;
        _windowWristMoveFrames = 0;
      } else {
        _steadyStartMs ??= timestampMs;
      }
    } else {
      stepDelta = 0.0;
      _lastStepDelta = 0.0;
      _lastWristShift = 0.0;
      _steadyStartMs ??= timestampMs;
    }

    _window.add(sample);
    _motionWindow.add(motion);
    _window.removeWhere((s) => s.timestampMs < timestampMs - windowMs);
    while (_window.length > maxWindowFrames) {
      _window.removeAt(0);
    }
    _motionWindow.removeWhere(
      (s) => s.timestampMs < timestampMs - motionWindowMs,
    );

    // Release an existing latch once the user clearly transitions away from
    // that letter's handshape (or briefly relaxes it across two frames).
    if (_latchedLetter != null && !isDynamicLetterActive) {
      final latchedIdx = classifier.labels.indexOf(_latchedLetter!);
      if (latchedIdx >= 0) {
        final twoLowFrames = _window.length >= 2 &&
            _window[_window.length - 1].probs[latchedIdx] < 0.36 &&
            _window[_window.length - 2].probs[latchedIdx] < 0.36;
        final switchedCleanly = sample.bestLabel != _latchedLetter &&
            sample.bestScore >= 0.75 &&
            sample.probs[latchedIdx] < 0.15 &&
            stepDelta < maxStepDisplacement;
        var lowWindowAverage = false;
        if (_window.length >= minFrames) {
          var sumLatched = 0.0;
          for (final s in _window) {
            sumLatched += s.probs[latchedIdx];
          }
          lowWindowAverage = sumLatched / _window.length < 0.35;
        }
        if (twoLowFrames || lowWindowAverage || switchedCleanly) {
          final previousLatched = _latchedLetter;
          _latchedLetter = null;
          activePrediction = null;
          isDynamicLetterActive = false;
          while (_window.isNotEmpty &&
              _window.first.bestLabel == previousLatched) {
            _window.removeAt(0);
          }
        }
      }
    }

    if (sample.bestScore >= classifier.threshold &&
        stepDelta < maxStepDisplacement &&
        !isDynamicLetterActive &&
        sample.bestLabel != 'J' &&
        sample.bestLabel != 'Z') {
      activePrediction = SignPrediction(
        sample.bestLabel,
        sample.bestScore,
        classifier.threshold,
      );
      retry = false;
      isDynamicLetterActive = false;
    }

    // Check dynamic trajectory patterns for J and Z before requiring complete
    // multi-frame static stillness.
    final dynamicMatch = _detectDynamicLetter();
    if (dynamicMatch != null) {
      final isFreshDynamicRepeat =
          timestampMs - (_lastDynamicLetterMs ?? -1000) >= 350;
      if (_latchedLetter != dynamicMatch.label || isFreshDynamicRepeat) {
        letters.add(dynamicMatch.label);
        _latchedLetter = dynamicMatch.label;
        _lastDynamicLetterMs = timestampMs;
      }
      activePrediction = dynamicMatch;
      retry = false;
      isDynamicLetterActive = true;
      _steadyStartMs = timestampMs;
      _window.clear();
      _motionWindow.clear();
      _lastStepDelta = 0.0;
      _lastWristShift = 0.0;
      _windowMeanStep = 0.0;
      _windowWristDrift = 0.0;
      _windowWristPath = 0.0;
      _windowWristMoveFrames = 0;
      return;
    }

    if (_window.length < minFrames) return;
    final spanMs = _window.last.timestampMs - _window.first.timestampMs;
    if (spanMs < minHoldMs) return;

    var stepSum = 0.0;
    var indexPath = 0.0;
    var pinkyPath = 0.0;
    var recentIndexPath = 0.0;
    var recentPinkyPath = 0.0;
    final recentStart = math.max(1, _window.length - 3);
    var wristPath = 0.0;
    var wristMoveFrames = 0;
    var scaleSum = _window.first.palmScale;
    var lastArrivalJumpIdx = 0;
    for (var i = 1; i < _window.length; i++) {
      final prev = _window[i - 1];
      final curr = _window[i];
      final step = _normalizedDisplacement(prev, curr);
      stepSum += step;
      final pairScale = math.max((prev.palmScale + curr.palmScale) * 0.5, 1e-4);
      final wDx = curr.wristX - prev.wristX;
      final wDy = curr.wristY - prev.wristY;
      final wStep = math.sqrt(wDx * wDx + wDy * wDy) / pairScale;
      wristPath += wStep;
      if (wStep >= 0.07) wristMoveFrames++;
      if (step >= 0.16 || wStep >= 0.16) {
        lastArrivalJumpIdx = i;
      }
      final iSeg = math.sqrt(
        (curr.indexX - prev.indexX) * (curr.indexX - prev.indexX) +
            (curr.indexY - prev.indexY) * (curr.indexY - prev.indexY),
      );
      final pSeg = math.sqrt(
        (curr.pinkyX - prev.pinkyX) * (curr.pinkyX - prev.pinkyX) +
            (curr.pinkyY - prev.pinkyY) * (curr.pinkyY - prev.pinkyY),
      );
      indexPath += iSeg;
      pinkyPath += pSeg;
      if (i >= recentStart) {
        recentIndexPath += iSeg;
        recentPinkyPath += pSeg;
      }
      scaleSum += curr.palmScale;
    }
    final meanStep = stepSum / (_window.length - 1);
    final meanScale = math.max(scaleSum / _window.length, 1e-4);
    var driftStartIdx = 0;
    if (lastArrivalJumpIdx > 0 &&
        _window.length - lastArrivalJumpIdx >= minFrames &&
        _window.last.timestampMs - _window[lastArrivalJumpIdx].timestampMs >=
            minHoldMs) {
      driftStartIdx = lastArrivalJumpIdx;
    }
    final netWristDx = _window.last.wristX - _window[driftStartIdx].wristX;
    final netWristDy = _window.last.wristY - _window[driftStartIdx].wristY;
    final netWristDrift =
        math.sqrt(netWristDx * netWristDx + netWristDy * netWristDy) /
            meanScale;
    final recentRefIdx = recentStart - 1;
    final recentIndexDx = _window.last.indexX - _window[recentRefIdx].indexX;
    final recentIndexDy = _window.last.indexY - _window[recentRefIdx].indexY;
    final recentIndexSpan =
        math.sqrt(recentIndexDx * recentIndexDx + recentIndexDy * recentIndexDy);
    final recentPinkyDx = _window.last.pinkyX - _window[recentRefIdx].pinkyX;
    final recentPinkyDy = _window.last.pinkyY - _window[recentRefIdx].pinkyY;
    final recentPinkySpan =
        math.sqrt(recentPinkyDx * recentPinkyDx + recentPinkyDy * recentPinkyDy);
    _windowMeanStep = meanStep;
    _windowWristDrift = netWristDrift;
    _windowWristPath = wristPath;
    _windowWristMoveFrames = wristMoveFrames;
    if (meanStep >= maxStepDisplacement ||
        netWristDrift >= maxNetWristDrift ||
        _lastStepDelta >= 0.22 ||
        _lastWristShift >= 0.22 ||
        (wristMoveFrames >= 4 && wristPath >= 0.50)) {
      return;
    }

    final counts = <String, int>{};
    for (final s in _window) {
      counts[s.bestLabel] = (counts[s.bestLabel] ?? 0) + 1;
    }

    var winner = _window.last.bestLabel;
    var winnerVotes = 0;
    for (final entry in counts.entries) {
      if (entry.value > winnerVotes) {
        winner = entry.key;
        winnerVotes = entry.value;
      }
    }

    // J and Z are dynamic trajectories in FSL; never confirm them from a
    // stationary handshape, and wait while I/D is still actively moving at the
    // end of the window into a potential J/Z stroke (spatial span filters out
    // stationary fingertip jitter).
    if (winner == 'J' ||
        winner == 'Z' ||
        (winner == 'I' &&
            pinkyPath / meanScale >= 0.14 &&
            recentPinkyPath / meanScale >= 0.08 &&
            recentPinkySpan / meanScale >= 0.10) ||
        (winner == 'D' &&
            indexPath / meanScale >= 0.14 &&
            recentIndexPath / meanScale >= 0.08 &&
            recentIndexSpan / meanScale >= 0.10)) {
      return;
    }

    final voteRatio = winnerVotes / _window.length;
    final winnerIdx = classifier.labels.indexOf(winner);
    var probSum = 0.0;
    for (final s in _window) {
      probSum += s.probs[winnerIdx];
    }
    final meanScore = probSum / _window.length;

    if (voteRatio >= minMajorityRatio && meanScore >= classifier.threshold) {
      if (_latchedLetter != winner) {
        letters.add(winner);
        _latchedLetter = winner;
        _motionWindow.clear();
      }
      activePrediction = SignPrediction(winner, meanScore, classifier.threshold);
      retry = false;
      isDynamicLetterActive = false;
      return;
    }

    final steadyElapsed =
        _steadyStartMs == null ? 0 : timestampMs - _steadyStartMs!;
    if (_latchedLetter == null && steadyElapsed >= retryDelayMs) {
      retry = true;
      activePrediction = null;
      isDynamicLetterActive = false;
    }
  }

  void _handleMissingHand(int timestampMs) {
    _firstMissingMs ??= timestampMs;
    final missingMs = timestampMs - _firstMissingMs!;
    if (missingMs >= missingWindowResetMs) {
      _window.clear();
      _motionWindow.clear();
      _steadyStartMs = null;
      _lastStepDelta = 0.0;
      _lastWristShift = 0.0;
      _windowMeanStep = 0.0;
      _windowWristDrift = 0.0;
      _windowWristPath = 0.0;
      _windowWristMoveFrames = 0;
    }
    if (missingMs >= disappearClearMs) {
      _latchedLetter = null;
      _lastDynamicLetterMs = null;
      activePrediction = null;
      retry = false;
      isDynamicLetterActive = false;
    }
  }

  _PrimaryHand? _selectPrimaryHand(List<List<LandmarkPoint>> hands) {
    _PrimaryHand? chosen;
    for (final hand in hands) {
      if (hand.length < 21) continue;
      final wrist = hand.first;
      var scaleSum = 0.0;
      for (final idx in const <int>[5, 9, 13, 17]) {
        final dx = hand[idx].x - wrist.x;
        final dy = hand[idx].y - wrist.y;
        scaleSum += math.sqrt(dx * dx + dy * dy);
      }
      final palmScale = scaleSum / 4.0;
      if (palmScale < 1e-4) continue;
      if (chosen == null || wrist.y < chosen.landmarks.first.y) {
        chosen = _PrimaryHand(hand, palmScale);
      }
    }
    return chosen;
  }

  double _normalizedDisplacement(_AlphabetSample a, _AlphabetSample b) {
    var internalSum = 0.0;
    for (var k = 0; k < 21; k++) {
      final base = k * 3;
      final dx = b.features[base] - a.features[base];
      final dy = b.features[base + 1] - a.features[base + 1];
      internalSum += math.sqrt(dx * dx + dy * dy);
    }
    final internalMean = internalSum / 21.0;
    final meanScale = math.max((a.palmScale + b.palmScale) * 0.5, 1e-4);
    final wristDx = b.wristX - a.wristX;
    final wristDy = b.wristY - a.wristY;
    final wristShift =
        math.sqrt(wristDx * wristDx + wristDy * wristDy) / meanScale;
    return internalMean + 0.5 * wristShift;
  }

  List<_MotionSample>? _contiguousStrokeSlice(int aIdx, int bIdx) {
    if (_motionWindow.length < 6) return null;
    double pairProb(_MotionSample s) =>
        (aIdx >= 0 ? s.probs[aIdx] : 0.0) + (bIdx >= 0 ? s.probs[bIdx] : 0.0);

    var end = _motionWindow.length - 1;
    if (pairProb(_motionWindow[end]) < 0.20) return null;
    var start = end;
    var lowRun = 0;
    for (var i = end; i >= 0; i--) {
      if (pairProb(_motionWindow[i]) >= 0.12) {
        start = i;
        lowRun = 0;
      } else {
        lowRun++;
        if (lowRun >= 2) break;
      }
    }
    final slice = _motionWindow.sublist(start);
    if (slice.length < 6) return null;
    if (slice.last.timestampMs - slice.first.timestampMs < minHoldMs) {
      return null;
    }
    return slice;
  }

  SignPrediction? _detectDynamicLetter() {
    if (_motionWindow.length < 6) return null;
    final iIdx = classifier.labels.indexOf('I');
    final jIdx = classifier.labels.indexOf('J');
    final dIdx = classifier.labels.indexOf('D');
    final zIdx = classifier.labels.indexOf('Z');

    double sliceMeanScale(List<_MotionSample> slice) {
      var sum = 0.0;
      for (final s in slice) {
        sum += s.palmScale;
      }
      return math.max(sum / slice.length, 1e-4);
    }

    double sliceSupport(List<_MotionSample> slice, int aIdx, int bIdx) {
      var total = 0.0;
      for (final s in slice) {
        total += (aIdx >= 0 ? s.probs[aIdx] : 0.0) +
            (bIdx >= 0 ? s.probs[bIdx] : 0.0);
      }
      return total / slice.length;
    }

    bool sliceValidSupport(List<_MotionSample> slice, int aIdx, int bIdx) {
      var valid = 0;
      for (final s in slice) {
        final p = (aIdx >= 0 ? s.probs[aIdx] : 0.0) +
            (bIdx >= 0 ? s.probs[bIdx] : 0.0);
        if (p >= 0.35) valid++;
      }
      final n = slice.length;
      final last = (aIdx >= 0 ? slice[n - 1].probs[aIdx] : 0.0) +
          (bIdx >= 0 ? slice[n - 1].probs[bIdx] : 0.0);
      final prev = (aIdx >= 0 ? slice[n - 2].probs[aIdx] : 0.0) +
          (bIdx >= 0 ? slice[n - 2].probs[bIdx] : 0.0);
      return valid * 2 >= n && 0.5 * (last + prev) >= 0.30;
    }

    final jSlice = _contiguousStrokeSlice(iIdx, jIdx);
    final jSupport = jSlice == null ? 0.0 : sliceSupport(jSlice, iIdx, jIdx);
    final zSlice = _contiguousStrokeSlice(dIdx, zIdx);
    final zSupport = zSlice == null ? 0.0 : sliceSupport(zSlice, dIdx, zIdx);

    // Candidate check + pinky downward hook for J.
    if (jSlice != null &&
        jSupport >= 0.45 &&
        jSupport >= zSupport &&
        sliceValidSupport(jSlice, iIdx, jIdx)) {
      final meanScale = sliceMeanScale(jSlice);
      final refX = jSlice.first.wristX;
      final refY = jSlice.first.wristY;
      final px = <double>[
        for (final s in jSlice) (s.pinkyX - refX) / meanScale,
      ];
      final py = <double>[
        for (final s in jSlice) (s.pinkyY - refY) / meanScale,
      ];
      var minX = px.first;
      var maxX = px.first;
      var maxY = py.first;
      var peakYIdx = 0;
      var pinkyPath = 0.0;
      for (var i = 0; i < px.length; i++) {
        if (px[i] < minX) minX = px[i];
        if (px[i] > maxX) maxX = px[i];
        if (py[i] > maxY) {
          maxY = py[i];
          peakYIdx = i;
        }
        if (i > 0) {
          final dx = px[i] - px[i - 1];
          final dy = py[i] - py[i - 1];
          pinkyPath += math.sqrt(dx * dx + dy * dy);
        }
      }
      final chordDx = px.last - px.first;
      final chordDy = py.last - py.first;
      final chordDist = math.sqrt(chordDx * chordDx + chordDy * chordDy);
      var maxChordDev = 0.0;
      var turnIdx = peakYIdx.clamp(1, px.length - 2);
      if (chordDist > 1e-4) {
        for (var i = 1; i < px.length - 1; i++) {
          final dev = ((px[i] - px.first) * chordDy -
                      (py[i] - py.first) * chordDx)
                  .abs() /
              chordDist;
          if (dev > maxChordDev) {
            maxChordDev = dev;
            turnIdx = i;
          }
        }
      }
      final downwardDrop = maxY - py.first;
      final rangeX = maxX - minX;
      final hookDx = (px.last - px.first).abs();
      final hasHookUp = peakYIdx < px.length - 1 && (maxY - py.last) >= 0.02;
      final hasLateralSweep = (py[turnIdx] - py.first) >= 0.16 &&
          (px.last - px[turnIdx]).abs() >= 0.10 &&
          (py.last - py[turnIdx]) <= 0.40 * downwardDrop;
      if (peakYIdx >= 2 &&
          downwardDrop >= 0.22 &&
          rangeX >= 0.16 &&
          hookDx >= 0.12 &&
          pinkyPath >= 0.30 &&
          chordDist <= pinkyPath * 0.92 &&
          maxChordDev >= 0.05 &&
          (hasHookUp || hasLateralSweep)) {
        final score = math.max(0.85, math.min(0.99, jSupport));
        return SignPrediction('J', score, classifier.threshold);
      }
    }

    // Candidate check + index-finger 3-segment zigzag (2 horizontal reversals) for Z.
    if (zSlice != null &&
        zSupport >= 0.45 &&
        zSupport > jSupport &&
        sliceValidSupport(zSlice, dIdx, zIdx)) {
      final meanScale = sliceMeanScale(zSlice);
      final refX = zSlice.first.wristX;
      final refY = zSlice.first.wristY;
      final ix = <double>[
        for (final s in zSlice) (s.indexX - refX) / meanScale,
      ];
      final iy = <double>[
        for (final s in zSlice) (s.indexY - refY) / meanScale,
      ];
      var minX = ix.first;
      var maxX = ix.first;
      var minY = iy.first;
      var maxY = iy.first;
      var indexPath = 0.0;
      var dir = 0;
      var accumDx = 0.0;
      var reversals = 0;
      int? turnCandidateIdx;
      var turnCandidateX = ix.first;
      final cornerIndices = <int>[];
      for (var i = 0; i < ix.length; i++) {
        if (ix[i] < minX) minX = ix[i];
        if (ix[i] > maxX) maxX = ix[i];
        if (iy[i] < minY) minY = iy[i];
        if (iy[i] > maxY) maxY = iy[i];
        if (i > 0) {
          final dx = ix[i] - ix[i - 1];
          final dy = iy[i] - iy[i - 1];
          indexPath += math.sqrt(dx * dx + dy * dy);
          if (dir == 0) {
            accumDx += dx;
            if (accumDx.abs() >= 0.08) {
              dir = accumDx > 0 ? 1 : -1;
              accumDx = 0.0;
              turnCandidateIdx = i;
              turnCandidateX = ix[i];
            }
          } else if (dx * dir > 0) {
            accumDx = 0.0;
            if (turnCandidateIdx == null ||
                (dir > 0 ? ix[i] > turnCandidateX : ix[i] < turnCandidateX)) {
              turnCandidateX = ix[i];
              turnCandidateIdx = i;
            }
          } else if (dx * dir < 0) {
            accumDx += dx;
            if (accumDx.abs() >= 0.08) {
              if (turnCandidateIdx != null) {
                cornerIndices.add(turnCandidateIdx);
              }
              dir = -dir;
              accumDx = 0.0;
              turnCandidateIdx = i;
              turnCandidateX = ix[i];
              reversals++;
            }
          }
        }
      }
      final rangeX = maxX - minX;
      final rangeY = maxY - minY;
      var hasDownwardDiagonal = false;
      for (var k = 0; k + 1 < cornerIndices.length; k++) {
        final c0 = cornerIndices[k];
        final c1 = cornerIndices[k + 1];
        final topSegStart = k > 0 ? cornerIndices[k - 1] : 0;
        final botSegEnd = k + 2 < cornerIndices.length
            ? cornerIndices[k + 2]
            : ix.length - 1;
        final topY = (iy[topSegStart] + iy[c0]) * 0.5;
        final botY = (iy[c1] + iy[botSegEnd]) * 0.5;
        if ((iy[c1] - iy[c0]) >= math.max(0.08, 0.35 * rangeY) &&
            (botY - topY) >= 0.08 &&
            (ix[c0] - ix[topSegStart]).abs() >= 0.14 &&
            (ix[botSegEnd] - ix[c1]).abs() >= 0.14) {
          hasDownwardDiagonal = true;
          break;
        }
      }
      if (indexPath >= 0.48 &&
          rangeX >= 0.20 &&
          rangeY >= 0.15 &&
          reversals >= 2 &&
          hasDownwardDiagonal) {
        final score = math.max(0.85, math.min(0.99, zSupport));
        return SignPrediction('Z', score, classifier.threshold);
      }
    }

    return null;
  }
}

class _PrimaryHand {
  const _PrimaryHand(this.landmarks, this.palmScale);
  final List<LandmarkPoint> landmarks;
  final double palmScale;
}

class _AlphabetSample {
  const _AlphabetSample({
    required this.timestampMs,
    required this.wristX,
    required this.wristY,
    required this.palmScale,
    required this.indexX,
    required this.indexY,
    required this.pinkyX,
    required this.pinkyY,
    required this.features,
    required this.probs,
    required this.bestLabel,
    required this.bestScore,
  });

  final int timestampMs;
  final double wristX;
  final double wristY;
  final double palmScale;
  final double indexX;
  final double indexY;
  final double pinkyX;
  final double pinkyY;
  final Float64List features;
  final Float64List probs;
  final String bestLabel;
  final double bestScore;
}

class _MotionSample {
  const _MotionSample({
    required this.timestampMs,
    required this.wristX,
    required this.wristY,
    required this.palmScale,
    required this.indexX,
    required this.indexY,
    required this.pinkyX,
    required this.pinkyY,
    required this.probs,
  });

  final int timestampMs;
  final double wristX;
  final double wristY;
  final double palmScale;
  final double indexX;
  final double indexY;
  final double pinkyX;
  final double pinkyY;
  final Float64List probs;
}
