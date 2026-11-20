import 'dart:math' as math;

import 'sign_classifier.dart';

/// Continuous dynamic sign capture. Completes when hands pause outside the
/// frame, after holding the ending posture of a movement steady, or at the max
/// clip duration, then waits for posture release before arming the next sign.
class SignCapture {
  SignCapture(this.classifier);
  final SignClassifier classifier;
  final List<SignFrame> _frames = [];
  final List<String> labels = <String>[];
  int? _start, _lastHand;
  SignFrame? _anchor, _departure;
  int _anchorTime = 0, _anchorIndex = 0, _departureTime = 0;
  double _maxTravel = 0.0;
  double _totalPath = 0.0;
  double _lastStep = 0.0;
  double _wristPath = 0.0;
  double _wristMaxTravel = 0.0;
  int _movingFrames = 0;
  int _consecutiveMovingFrames = 0;
  int _framesSinceLastMove = 0;
  int _anchorShifts = 0;
  bool _sawTwoHands = false;
  bool _lastFinishWasDynamic = false;
  bool _waitingForRelease = false;
  SignPrediction? result;
  bool retry = false;
  bool get recording => _frames.isNotEmpty;
  bool get isWaitingForRelease => _waitingForRelease;
  double get maxTravel => _maxTravel;
  double get totalPath => _totalPath;
  double get lastStep => _lastStep;
  double get wristMaxTravel => _wristMaxTravel;
  double get wristPath => _wristPath;
  int get movingFrames => _movingFrames;
  int get anchorShifts => _anchorShifts;
  bool get sawTwoHands => _sawTwoHands;
  bool get lastFinishWasDynamic => _lastFinishWasDynamic;
  bool get isSettledInPlaceMorph =>
      recording &&
      !_sawTwoHands &&
      _wristMaxTravel < 0.40 &&
      _wristPath < 0.55 &&
      _framesSinceLastMove >= 2;
  bool get isCurrentlyMoving => recording && _lastStep > 0.15;
  bool get hasActiveMovement =>
      recording &&
      (_sawTwoHands ||
          isCurrentlyMoving ||
          _consecutiveMovingFrames >= 2 ||
          (!isSettledInPlaceMorph &&
              (_movingFrames >= 4 ||
                  (_totalPath >= 0.85 &&
                      (_maxTravel >= 0.35 || _wristPath >= 0.35)) ||
                  _maxTravel >= 0.85)));
  bool get isDynamicGestureCandidate =>
      recording &&
      (_sawTwoHands ||
          _movingFrames >= 3 ||
          (_totalPath >= 1.25 && (_maxTravel >= 0.40 || _wristPath >= 0.45)) ||
          _maxTravel >= 0.85 ||
          (_anchorShifts >= 2 && _totalPath >= 0.65));

  void latchStaticPosture(SignFrame hands, int milliseconds) {
    _frames.clear();
    _resetTracking();
    _lastFinishWasDynamic = false;
    result = null;
    retry = false;
    if (hands.isNotEmpty) {
      _lastHand = milliseconds;
      _anchor = hands;
      _anchorTime = milliseconds;
      _waitingForRelease = true;
    } else {
      _waitingForRelease = false;
    }
  }

  void resetSegment() {
    _frames.clear();
    _resetTracking();
    _lastFinishWasDynamic = false;
    _start = null;
    _lastHand = null;
    _waitingForRelease = false;
    result = null;
    retry = false;
  }

  void reset() {
    resetSegment();
    labels.clear();
  }

  void add(SignFrame hands, int milliseconds) {
    if (_lastHand != null && milliseconds < _lastHand!) {
      resetSegment();
    }
    var finishedInCall = false;
    final gap = milliseconds - (_lastHand ?? milliseconds);
    if (_waitingForRelease && gap >= 160) {
      _resetTracking();
      _waitingForRelease = false;
    } else if (gap >= 280) {
      if (recording) {
        _finish(endedByGap: true);
        finishedInCall = true;
      }
      _resetTracking();
      _waitingForRelease = false;
    }
    if (hands.isEmpty) return;
    _lastHand = milliseconds;
    if (_waitingForRelease) {
      final anchor = _anchor;
      if (anchor == null) {
        _resetTracking();
        _waitingForRelease = false;
      } else if (_distance(anchor, hands) <= 0.40) {
        _departure = null;
        return;
      } else if (_departure == null) {
        _departure = hands;
        _departureTime = milliseconds;
        return;
      } else if (milliseconds - _departureTime < 45) {
        return;
      } else {
        final firstFrame = _departure!;
        final firstTime = _departureTime;
        _resetTracking();
        _waitingForRelease = false;
        _lastFinishWasDynamic = false;
        _start = _anchorTime = firstTime;
        _anchor = firstFrame;
        _anchorIndex = 0;
        _frames.add(firstFrame);
        _sawTwoHands = firstFrame.length >= 2;
        result = null;
        retry = false;
      }
    }
    if (!recording) {
      _start = _anchorTime = milliseconds;
      _anchor = hands;
      _anchorIndex = 0;
      _departure = null;
      _maxTravel = 0.0;
      _totalPath = 0.0;
      _lastStep = 0.0;
      _wristPath = 0.0;
      _wristMaxTravel = 0.0;
      _movingFrames = 0;
      _consecutiveMovingFrames = 0;
      _framesSinceLastMove = 0;
      _anchorShifts = 0;
      _sawTwoHands = hands.length >= 2;
      if (!finishedInCall) {
        _lastFinishWasDynamic = false;
        result = null;
        retry = false;
      }
    }
    _frames.add(hands);
    if (hands.length >= 2) _sawTwoHands = true;
    if (_frames.length > 1) {
      final prev = _frames[_frames.length - 2];
      if (prev.length == hands.length) {
        final step = _distance(prev, hands);
        final wStep = _wristDistance(prev, hands);
        if (wStep.isFinite) {
          _wristPath += wStep;
        }
        if (step.isFinite) {
          _lastStep = step;
          _totalPath += step;
          if (step > 0.14) {
            _movingFrames++;
            _consecutiveMovingFrames++;
            _framesSinceLastMove = 0;
          } else {
            _consecutiveMovingFrames = 0;
            _framesSinceLastMove++;
          }
        } else {
          _lastStep = 0.0;
          _consecutiveMovingFrames = 0;
          _framesSinceLastMove++;
        }
      } else {
        _lastStep = 0.0;
        _consecutiveMovingFrames = 0;
        _framesSinceLastMove++;
      }
      if (_frames.first.length == hands.length) {
        final excursion = _distance(_frames.first, hands);
        if (excursion.isFinite && excursion > _maxTravel) {
          _maxTravel = excursion;
        }
        final wristExc = _wristDistance(_frames.first, hands);
        if (wristExc.isFinite && wristExc > _wristMaxTravel) {
          _wristMaxTravel = wristExc;
        }
      }
    } else {
      _lastStep = 0.0;
      _consecutiveMovingFrames = 0;
      _framesSinceLastMove = 0;
    }
    if (_anchor != null && _distance(_anchor!, hands) > 0.28) {
      _anchor = hands;
      _anchorTime = milliseconds;
      _anchorIndex = _frames.length - 1;
      _anchorShifts++;
    }
    final elapsed = milliseconds - _start!;
    if (_frames.length >= 6 &&
        elapsed >= 500 &&
        _anchorIndex >= 3 &&
        milliseconds - _anchorTime >= 750) {
      if (_isAnchorWindowSettled()) {
        if (_anchorIndex + 2 < _frames.length) {
          _frames.removeRange(_anchorIndex + 2, _frames.length);
        }
        _finish();
        _anchor = hands;
        _waitingForRelease = true;
        return;
      } else {
        _anchor = hands;
        _anchorTime = milliseconds;
        _anchorIndex = _frames.length - 1;
      }
    }
    if (elapsed >= 2900) {
      if (_anchorIndex >= 3 &&
          milliseconds - _anchorTime >= 300 &&
          _anchorIndex + 2 < _frames.length) {
        _frames.removeRange(_anchorIndex + 2, _frames.length);
      }
      _finish();
      _anchor = hands;
      _waitingForRelease = true;
    }
  }

  void _finish({bool endedByGap = false}) {
    final durationMs = (_lastHand != null && _start != null)
        ? (_lastHand! - _start!)
        : 0;
    final hasDynamicExtent = _sawTwoHands ||
        _movingFrames >= 3 ||
        (_totalPath >= 1.25 && (_maxTravel >= 0.40 || _wristPath >= 0.45)) ||
        _maxTravel >= 0.85 ||
        (_anchorShifts >= 2 && _totalPath >= 0.65);
    final isMonotonicDropExit =
        endedByGap && _isMonotonicDownwardExit(durationMs);
    _lastFinishWasDynamic = hasDynamicExtent && !isMonotonicDropExit;
    final score = _lastFinishWasDynamic &&
            _frames.length >= 5 &&
            durationMs >= 350
        ? classifier.predict(_frames)
        : null;
    result = score != null && classifier.accepts(score) ? score : null;
    if (result != null) {
      labels.add(result!.label);
    }
    retry = !isMonotonicDropExit && result == null;
    _frames.clear();
  }

  bool _isAnchorWindowSettled() {
    var path = 0.0;
    var moving = 0;
    for (var i = _anchorIndex + 1; i < _frames.length; i++) {
      final step = _distance(_frames[i - 1], _frames[i]);
      if (step.isFinite) {
        path += step;
        if (step > 0.14) moving++;
      }
    }
    return moving < 3 && path < 0.85;
  }

  bool _isMonotonicDownwardExit(int durationMs) {
    if (_sawTwoHands || _frames.length < 3 || _frames.length > 18) {
      return false;
    }
    if (durationMs > 750) return false;
    final firstHand = _frames.first.first;
    final lastHand = _frames.last.first;
    final palm = math.max(
      0.015,
      (math.sqrt(math.pow(firstHand[0].x - firstHand[9].x, 2) +
                  math.pow(firstHand[0].y - firstHand[9].y, 2)) +
              math.sqrt(math.pow(lastHand[0].x - lastHand[9].x, 2) +
                  math.pow(lastHand[0].y - lastHand[9].y, 2))) /
          2,
    );
    final netDy = lastHand[0].y - firstHand[0].y;
    final netDx = (lastHand[0].x - firstHand[0].x).abs();
    if (netDy < 0.30 * palm || netDx > 0.65 * netDy) return false;
    var minY = firstHand[0].y;
    var upwardDy = 0.0;
    for (var i = 1; i < _frames.length; i++) {
      if (_frames[i].length != 1 || _frames[i - 1].length != 1) return false;
      final y = _frames[i].first[0].y;
      if (y < minY) minY = y;
      final stepDy = y - _frames[i - 1].first[0].y;
      if (stepDy < 0) upwardDy += -stepDy;
    }
    final maxRise = firstHand[0].y - minY;
    if (maxRise > 0.25 * palm ||
        upwardDy > math.max(0.25 * palm, 0.25 * netDy)) {
      return false;
    }
    var shapeDeltaSum = 0.0;
    for (final idx in const [4, 8, 12, 16, 20]) {
      final ax = firstHand[idx].x - firstHand[0].x;
      final ay = firstHand[idx].y - firstHand[0].y;
      final bx = lastHand[idx].x - lastHand[0].x;
      final by = lastHand[idx].y - lastHand[0].y;
      shapeDeltaSum += math.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    }
    final meanShapeDelta = shapeDeltaSum / (5 * palm);
    return meanShapeDelta < 0.35;
  }

  void _resetTracking() {
    _anchor = null;
    _departure = null;
    _anchorTime = 0;
    _anchorIndex = 0;
    _departureTime = 0;
    _maxTravel = 0.0;
    _totalPath = 0.0;
    _lastStep = 0.0;
    _wristPath = 0.0;
    _wristMaxTravel = 0.0;
    _movingFrames = 0;
    _consecutiveMovingFrames = 0;
    _framesSinceLastMove = 0;
    _anchorShifts = 0;
    _sawTwoHands = false;
  }

  static double _wristDistance(SignFrame a, SignFrame b) {
    if (a.length != b.length) return double.infinity;
    double pairDist(List<SignPoint> p, List<SignPoint> q) {
      final palmA = math.sqrt(
        math.pow(p.first.x - p[9].x, 2) + math.pow(p.first.y - p[9].y, 2),
      );
      final palmB = math.sqrt(
        math.pow(q.first.x - q[9].x, 2) + math.pow(q.first.y - q[9].y, 2),
      );
      final palm = math.max(0.015, (palmA + palmB) / 2);
      final dx = p.first.x - q.first.x;
      final dy = p.first.y - q.first.y;
      return math.sqrt(dx * dx + dy * dy) / palm;
    }

    if (a.length == 1) return pairDist(a.first, b.first);
    final direct = math.max(pairDist(a[0], b[0]), pairDist(a[1], b[1]));
    final swapped = math.max(pairDist(a[0], b[1]), pairDist(a[1], b[0]));
    return math.min(direct, swapped);
  }

  static double _handDistance(List<SignPoint> a, List<SignPoint> b) {
    double distance(SignPoint p, SignPoint q) {
      final x = p.x - q.x, y = p.y - q.y;
      return math.sqrt(x * x + y * y);
    }

    final palm = (distance(a.first, a[9]) + distance(b.first, b[9])) / 2;
    var sum = 0.0;
    for (final i in const [0, 4, 8, 12, 16, 20]) {
      sum += distance(a[i], b[i]);
    }
    return sum / (6 * math.max(0.015, palm));
  }

  static double _distance(SignFrame a, SignFrame b) {
    if (a.length != b.length) return double.infinity;
    if (a.length == 1) return _handDistance(a.first, b.first);
    final direct =
        math.max(_handDistance(a[0], b[0]), _handDistance(a[1], b[1]));
    final swapped =
        math.max(_handDistance(a[0], b[1]), _handDistance(a[1], b[0]));
    return math.min(direct, swapped);
  }
}
