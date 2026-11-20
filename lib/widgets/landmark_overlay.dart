import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';

import '../services/landmark_service.dart';

/// Display-only. Raw detections continue to the sign capture pipeline while
/// [displayHands], [displayPose], and [statusRevision] drive low-latency,
/// jitter-free overlay painting and rebuild-free status indicators.
class LandmarkOverlay extends ChangeNotifier {
  static const dropoutGrace = Duration(milliseconds: 250);
  final _tracks = [_HandTrack(), _HandTrack()];
  List<List<LandmarkPoint>> get hands => [
        for (final track in _tracks)
          if (track.points.isNotEmpty) track.points
      ];
  List<List<LandmarkPoint>> get displayHands => [
        for (final track in _tracks)
          if (track.points.isNotEmpty) track.displayPoints
      ];

  List<LandmarkPoint> pose = const [];
  List<LandmarkPoint> _displayPose = const [];
  List<LandmarkPoint> get displayPose =>
      pose.isEmpty ? const [] : _displayPose;

  /// Increments only when hand or pose visibility transitions (present <-> empty),
  /// preventing full result-panel widget rebuilds on every camera frame.
  final ValueNotifier<int> statusRevision = ValueNotifier<int>(0);
  bool _lastHasHands = false;
  bool _lastHasPose = false;

  Timer? _poseExpiry;

  void update(LandmarkFrame frame) {
    final incoming = frame.hands.where((h) => h.isNotEmpty).take(2).toList();
    // Match the nearest wrists first. A one-frame miss on either hand must
    // leave the other hand free to move, regardless of detector slot order.
    final free = _tracks.toList();
    while (incoming.isNotEmpty) {
      var track = free.first;
      var index = 0;
      var best = double.infinity;
      for (final candidate in free.where((t) => t.points.isNotEmpty)) {
        final wrist = candidate.points.first;
        for (var i = 0; i < incoming.length; i++) {
          final dx = incoming[i].first.x - wrist.x;
          final dy = incoming[i].first.y - wrist.y;
          final distance = dx * dx + dy * dy;
          if (distance < best) {
            best = distance;
            track = candidate;
            index = i;
          }
        }
      }
      track.expiry?.cancel();
      track.expiry = null;
      track.updatePoints(incoming.removeAt(index));
      free.remove(track);
    }
    for (final track in free.where((t) => t.points.isNotEmpty)) {
      track.expiry ??= Timer(dropoutGrace, () {
        track.clear();
        _syncStatus();
        notifyListeners();
      });
    }
    if (frame.pose.isNotEmpty) {
      _poseExpiry?.cancel();
      _poseExpiry = null;
      pose = frame.pose;
      _updateDisplayPose(frame.pose);
    } else if (frame.poseSampled && pose.isNotEmpty) {
      _poseExpiry ??= Timer(dropoutGrace, () {
        pose = const [];
        _displayPose = const [];
        _poseExpiry = null;
        _syncStatus();
        notifyListeners();
      });
    } else if (_displayPose.length >= 7 &&
        _tracks.any((t) => t.points.isNotEmpty && t.lastWristDelta != null)) {
      _propagateHandMotionToDisplayPose(frame);
    }
    _syncStatus();
    notifyListeners();
  }

  void _updateDisplayPose(List<LandmarkPoint> target) {
    if (_displayPose.length != target.length || target.length < 3) {
      _displayPose = List<LandmarkPoint>.of(target, growable: false);
      return;
    }
    final shoulderSpan = math.max(
      0.06,
      _dist2d(target[2].x - target[1].x, target[2].y - target[1].y),
    );
    var dispSum = 0.0;
    final sampleCount = math.min(5, target.length);
    for (var i = 0; i < sampleCount; i++) {
      dispSum += _dist2d(
        target[i].x - _displayPose[i].x,
        target[i].y - _displayPose[i].y,
      );
    }
    final avgNormDisp = dispSum / (sampleCount * shoulderSpan);
    if (avgNormDisp > 0.45) {
      _displayPose = List<LandmarkPoint>.of(target, growable: false);
      return;
    }
    final alpha = avgNormDisp <= 0.015
        ? 0.46
        : (0.46 + (avgNormDisp - 0.015) / 0.065 * 0.48).clamp(0.46, 0.94);
    _displayPose = List<LandmarkPoint>.generate(target.length, (i) {
      final p = _displayPose[i];
      final n = target[i];
      return LandmarkPoint(
        p.x + (n.x - p.x) * alpha,
        p.y + (n.y - p.y) * alpha,
        p.z + (n.z - p.z) * alpha,
      );
    }, growable: false);
  }

  void _propagateHandMotionToDisplayPose(LandmarkFrame frame) {
    if (pose.isEmpty || _displayPose.length < 7 || frame.rotation != 0) return;
    final next = List<LandmarkPoint>.of(_displayPose, growable: false);
    var changed = false;
    const maxMatchDist = 0.22;
    for (final track in _tracks) {
      final delta = track.lastWristDelta;
      if (track.points.isEmpty || delta == null) continue;
      if (delta.dx.abs() < 1e-4 && delta.dy.abs() < 1e-4) continue;
      final handWrist = track.displayPoints.first;
      var bestIdx = -1;
      var bestDist = maxMatchDist;
      for (final wristIdx in const <int>[5, 6]) {
        final d = _dist2d(
          handWrist.x - next[wristIdx].x,
          handWrist.y - next[wristIdx].y,
        );
        if (d < bestDist) {
          bestDist = d;
          bestIdx = wristIdx;
        }
      }
      if (bestIdx != -1) {
        final elbowIdx = bestIdx == 5 ? 3 : 4;
        next[bestIdx] = LandmarkPoint(handWrist.x, handWrist.y, handWrist.z);
        next[elbowIdx] = LandmarkPoint(
          (next[elbowIdx].x + delta.dx * 0.28).clamp(0.0, 1.0),
          (next[elbowIdx].y + delta.dy * 0.28).clamp(0.0, 1.0),
          next[elbowIdx].z,
        );
        changed = true;
      }
    }
    if (changed) {
      _displayPose = next;
    }
  }

  void _syncStatus() {
    final hasHands = _tracks.any((t) => t.points.isNotEmpty);
    final hasPose = pose.isNotEmpty;
    if (hasHands != _lastHasHands || hasPose != _lastHasPose) {
      _lastHasHands = hasHands;
      _lastHasPose = hasPose;
      statusRevision.value++;
    }
  }

  void reset() {
    _clear();
    _syncStatus();
    notifyListeners();
  }

  void _clear() {
    for (final track in _tracks) {
      track.clear();
    }
    _poseExpiry?.cancel();
    _poseExpiry = null;
    pose = const [];
    _displayPose = const [];
  }

  @override
  void dispose() {
    _clear();
    statusRevision.dispose();
    super.dispose();
  }
}

class _HandTrack {
  List<LandmarkPoint> points = const [];
  List<LandmarkPoint> displayPoints = const [];
  ({double dx, double dy})? lastWristDelta;
  Timer? expiry;

  void updatePoints(List<LandmarkPoint> next) {
    final prevDisplay = displayPoints;
    points = next;
    if (prevDisplay.length != next.length || next.isEmpty) {
      displayPoints = List<LandmarkPoint>.of(next, growable: false);
      lastWristDelta = null;
      return;
    }
    final palm = next.length > 9
        ? math.max(
            0.03,
            _dist2d(next[9].x - next[0].x, next[9].y - next[0].y),
          )
        : 0.12;
    var dispSum = 0.0;
    var count = 0;
    for (final idx in const <int>[0, 4, 8, 9, 12, 16, 20]) {
      if (idx >= next.length) continue;
      dispSum += _dist2d(
        next[idx].x - prevDisplay[idx].x,
        next[idx].y - prevDisplay[idx].y,
      );
      count++;
    }
    final avgNormDisp = count > 0 ? dispSum / (count * palm) : 1.0;
    if (avgNormDisp > 0.36) {
      displayPoints = List<LandmarkPoint>.of(next, growable: false);
      lastWristDelta = (
        dx: next.first.x - prevDisplay.first.x,
        dy: next.first.y - prevDisplay.first.y,
      );
      return;
    }
    final alpha = avgNormDisp <= 0.012
        ? 0.44
        : (0.44 + (avgNormDisp - 0.012) / 0.058 * 0.51).clamp(0.44, 0.95);
    displayPoints = List<LandmarkPoint>.generate(next.length, (i) {
      final p = prevDisplay[i];
      final n = next[i];
      return LandmarkPoint(
        p.x + (n.x - p.x) * alpha,
        p.y + (n.y - p.y) * alpha,
        p.z + (n.z - p.z) * alpha,
      );
    }, growable: false);
    lastWristDelta = (
      dx: displayPoints.first.x - prevDisplay.first.x,
      dy: displayPoints.first.y - prevDisplay.first.y,
    );
  }

  void clear() {
    expiry?.cancel();
    expiry = null;
    points = const [];
    displayPoints = const [];
    lastWristDelta = null;
  }
}

double _dist2d(double dx, double dy) => math.sqrt(dx * dx + dy * dy);
