import 'dart:math' as math;

import 'sign_classifier.dart';

/// Trial mode: a held final posture closes one sign. Measured from an anchor,
/// so slow travel accumulates, while small camera jitter does not rearm it.
class ContinuousSignCapture {
  ContinuousSignCapture(this.model);
  final SignClassifier model;
  final List<String> labels = [];
  final List<SignFrame> _frames = [];
  SignFrame? _anchor;
  int _anchorTime = 0, _start = 0, _lastSeen = 0, _anchorIndex = 0;
  bool _latched = false;
  SignFrame? _departure;
  int _departureTime = 0;
  bool retry = false;

  void resetSegment() {
    _frames.clear();
    _anchor = null;
    _latched = false;
    retry = false;
  }

  void add(SignFrame hands, int now) {
    if (_anchor != null && now - _lastSeen >= 750) {
      if (!_latched) _finish();
      _anchor = null;
      _latched = false;
    }
    if (hands.isEmpty) return;
    _lastSeen = now;
    if (_anchor == null) {
      _begin(hands, now);
      return;
    }
    final moved = _distance(_anchor!, hands) > .32;
    if (_latched) {
      if (_distance(_anchor!, hands) <= .48) {
        _departure = null;
      } else if (_departure == null) {
        _departure = hands;
        _departureTime = now;
      } else if (now - _departureTime >= 100) {
        _begin(_departure!, _departureTime);
        _frames.add(hands);
        _anchor = hands;
        _anchorTime = now;
      }

      return;
    }
    _frames.add(hands);
    if (moved) {
      _anchor = hands;
      _anchorTime = now;
      _anchorIndex = _frames.length - 1;
    }
    if (now - _anchorTime >= 900 &&
        now - _start >= 950 &&
        _frames.length >= 5) {
      // A long hold should count once, and must not dominate the motion clip.
      if (_anchorIndex >= 4) {
        _frames.removeRange(_anchorIndex + 1, _frames.length);
      }
      _finish();
      _latched = true;
    } else if (now - _start > 6500) {
      _frames.clear();
      retry = true;
      // Wait for a pause or release before starting another segment.
      _latched = false;
      _start = now;
      _anchorIndex = 0;
      _discardUntilPause = true;
    }
  }

  bool _discardUntilPause = false;

  void _begin(SignFrame hands, int now) {
    _frames.clear();
    _departure = null;
    _frames.add(hands);
    _start = _anchorTime = now;
    _anchorIndex = 0;
    _anchor = hands;
    _latched = false;
    _discardUntilPause = false;
    retry = false;
  }

  void _finish() {
    final result =
        !_discardUntilPause && _frames.length >= 5 && _lastSeen - _start >= 300
            ? model.predict(_frames)
            : null;
    if (result != null && model.accepts(result)) {
      labels.add(result.label);
      retry = false;
    } else {
      retry = true;
    }
    _frames.clear();
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
    return sum / (6 * math.max(.015, palm));
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
