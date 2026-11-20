import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../services/app_preferences.dart';
import '../theme/visualin_tokens.dart';
import 'hand_landmark_connections.dart';

/// Paints either the modern cyan/teal anatomical contour tracking overlay
/// ([TrackingOverlayStyle.contour]) or the classic orange bone-and-joint
/// skeleton overlay ([TrackingOverlayStyle.skeleton]) in preview canvas space.
class LandmarkContourRenderer {
  const LandmarkContourRenderer._();

  // Cached Paint instances eliminate per-frame heap allocations during 20-30fps
  // Canvas repaints on Android.
  static final Paint _handHaloPaint = Paint()
    ..color = const Color(0x4214B8CC)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 3.8
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _handCorePaint = Paint()
    ..color = const Color(0xFF3BF0FF)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.65
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _handHighContrastPaint = Paint()
    ..color = Colors.white
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.3
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _armOuterPaint = Paint()
    ..color = const Color(0x2814B8CC)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.6
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _armLinePaint = Paint()
    ..color = const Color(0x7819D7EC)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.25
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _armHighContrastPaint = Paint()
    ..color = const Color(0xCC66F7FF)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.8
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _torsoContourPaint = Paint()
    ..color = const Color(0x4219D7EC)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.25
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _torsoHighContrastPaint = Paint()
    ..color = const Color(0x9966F7FF)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.7
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _headHairlinePaint = Paint()
    ..color = const Color(0x2E19D7EC)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.0;

  static final Paint _headCornerHaloPaint = Paint()
    ..color = const Color(0x4214B8CC)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 3.8
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _headCornerPaint = Paint()
    ..color = const Color(0xE642F5FF)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.0
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _headHighContrastCornerPaint = Paint()
    ..color = Colors.white
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.4
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _jointRingHaloPaint = Paint()
    ..color = const Color(0x3819D7EC)
    ..style = PaintingStyle.fill;

  static final Paint _jointRingStrokePaint = Paint()
    ..color = const Color(0xC23BF0FF)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 1.35;

  static final Paint _indicatorCorePaint = Paint()
    ..color = const Color(0xFFE8FEFF)
    ..style = PaintingStyle.fill;

  static final Paint _fingertipRingPaint = Paint()
    ..color = const Color(0x6642F5FF)
    ..style = PaintingStyle.fill;

  static final Paint _skeletonLinePaint = Paint()
    ..color = const Color(0xB3FF6B0A)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.0
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;

  static final Paint _skeletonJointPaint = Paint()
    ..color = VisualinColors.primaryOrange
    ..style = PaintingStyle.fill;

  static final Paint _skeletonFingertipRingPaint = Paint()
    ..color = const Color(0x4DFF6B0A)
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.4;

  /// Paints the selected overlay [style] from transformed canvas points.
  static void paint({
    required Canvas canvas,
    required Size size,
    required List<List<Offset>> hands,
    required List<Offset> posePoints,
    required TrackingOverlayStyle style,
    bool highContrast = false,
  }) {
    if (style == TrackingOverlayStyle.skeleton) {
      _paintSkeleton(canvas, hands: hands, posePoints: posePoints);
      return;
    }
    _paintContours(
      canvas,
      size: size,
      hands: hands,
      posePoints: posePoints,
      highContrast: highContrast,
    );
  }

  static void _paintSkeleton(
    Canvas canvas, {
    required List<List<Offset>> hands,
    required List<Offset> posePoints,
  }) {
    if (posePoints.length >= 5) {
      for (final edge in const [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6)]) {
        if (edge.$2 >= posePoints.length) continue;
        canvas.drawLine(
          posePoints[edge.$1],
          posePoints[edge.$2],
          _skeletonLinePaint,
        );
      }
      final bodyJointLimit = math.min(7, posePoints.length);
      for (var i = 0; i < bodyJointLimit; i++) {
        canvas.drawCircle(posePoints[i], 3.5, _skeletonJointPaint);
      }
    }
    for (final points in hands) {
      for (final connection in handLandmarkConnections) {
        if (connection.startIndex >= points.length ||
            connection.endIndex >= points.length) {
          continue;
        }
        canvas.drawLine(
          points[connection.startIndex],
          points[connection.endIndex],
          _skeletonLinePaint,
        );
      }
      for (var i = 0; i < points.length; i++) {
        final isFingertip = handLandmarkFingertips.contains(i);
        if (isFingertip) {
          canvas.drawCircle(points[i], 7.0, _skeletonFingertipRingPaint);
        }
        canvas.drawCircle(
          points[i],
          isFingertip ? 5.5 : 3.5,
          _skeletonJointPaint,
        );
      }
    }
  }

  static void _paintContours(
    Canvas canvas, {
    required Size size,
    required List<List<Offset>> hands,
    required List<Offset> posePoints,
    required bool highContrast,
  }) {
    final shoulderWidth = posePoints.length >= 3
        ? (posePoints[2] - posePoints[1]).distance
        : 0.0;

    if (posePoints.length >= 5) {
      _paintTorsoContour(
        canvas,
        size: size,
        posePoints: posePoints,
        shoulderWidth: shoulderWidth,
        highContrast: highContrast,
      );
      _paintArmContours(
        canvas,
        posePoints: posePoints,
        hands: hands,
        shoulderWidth: shoulderWidth,
        highContrast: highContrast,
      );
      _paintHeadFrame(
        canvas,
        size: size,
        posePoints: posePoints,
        highContrast: highContrast,
      );
    }

    for (final hand in hands) {
      _paintHandContour(canvas, hand, highContrast: highContrast);
    }
  }

  /// Computes a clean, non-obstructive head tracking bounding rect using nose,
  /// ears (when available from 11-point pose), and proportional shoulder width.
  static Rect? computeHeadTrackingRect(List<Offset> posePoints, Size size) {
    if (posePoints.isEmpty) return null;
    final nose = posePoints[0];
    final hasShoulders = posePoints.length >= 3;
    final shoulderW = hasShoulders
        ? (posePoints[2] - posePoints[1]).distance.clamp(32.0, size.width * 0.9)
        : size.shortestSide * 0.34;

    double width;
    double cx;
    double cy;

    if (posePoints.length >= 9) {
      final earL = posePoints[7];
      final earR = posePoints[8];
      final earSpan = (earR - earL).distance;
      if (earSpan > 6.0) {
        final earMid = (earL + earR) * 0.5;
        final noseEarDist = (nose - earMid).distance;
        final rawWidth = math.max(earSpan * 1.42, noseEarDist * 2.15);
        width = hasShoulders
            ? rawWidth.clamp(
                math.max(40.0, shoulderW * 0.36),
                math.max(44.0, shoulderW * 0.72),
              )
            : rawWidth.clamp(40.0, size.shortestSide * 0.60);
        cx = earMid.dx * 0.60 + nose.dx * 0.40;
        final height = width * 1.24;
        cy = (earMid.dy * 0.52 + nose.dy * 0.48) - height * 0.06;
        return Rect.fromCenter(
          center: Offset(
            cx.clamp(width * 0.25, size.width - width * 0.25),
            cy.clamp(height * 0.25, size.height - height * 0.25),
          ),
          width: width,
          height: height,
        );
      }
    }

    width = (shoulderW * 0.46).clamp(42.0, size.shortestSide * 0.52);
    final height = width * 1.24;
    cx = nose.dx.clamp(width * 0.25, size.width - width * 0.25);
    cy = (nose.dy - height * 0.07)
        .clamp(height * 0.25, size.height - height * 0.25);
    return Rect.fromCenter(center: Offset(cx, cy), width: width, height: height);
  }

  static void _paintHeadFrame(
    Canvas canvas, {
    required Size size,
    required List<Offset> posePoints,
    required bool highContrast,
  }) {
    final rect = computeHeadTrackingRect(posePoints, size);
    if (rect == null) return;

    final cornerRadius = (rect.width * 0.20).clamp(10.0, 22.0);
    final rrect = RRect.fromRectAndRadius(rect, Radius.circular(cornerRadius));
    canvas.drawRRect(rrect, _headHairlinePaint);

    final leg = (rect.shortestSide * 0.21).clamp(9.0, 20.0);
    final cornerPath = Path()
      // Top-left
      ..moveTo(rect.left, rect.top + leg)
      ..lineTo(rect.left, rect.top + cornerRadius)
      ..arcToPoint(
        Offset(rect.left + cornerRadius, rect.top),
        radius: Radius.circular(cornerRadius),
      )
      ..lineTo(rect.left + leg, rect.top)
      // Top-right
      ..moveTo(rect.right - leg, rect.top)
      ..lineTo(rect.right - cornerRadius, rect.top)
      ..arcToPoint(
        Offset(rect.right, rect.top + cornerRadius),
        radius: Radius.circular(cornerRadius),
      )
      ..lineTo(rect.right, rect.top + leg)
      // Bottom-right
      ..moveTo(rect.right, rect.bottom - leg)
      ..lineTo(rect.right, rect.bottom - cornerRadius)
      ..arcToPoint(
        Offset(rect.right - cornerRadius, rect.bottom),
        radius: Radius.circular(cornerRadius),
      )
      ..lineTo(rect.right - leg, rect.bottom)
      // Bottom-left
      ..moveTo(rect.left + leg, rect.bottom)
      ..lineTo(rect.left + cornerRadius, rect.bottom)
      ..arcToPoint(
        Offset(rect.left, rect.bottom - cornerRadius),
        radius: Radius.circular(cornerRadius),
      )
      ..lineTo(rect.left, rect.bottom - leg);

    if (!highContrast) {
      canvas.drawPath(cornerPath, _headCornerHaloPaint);
    }
    canvas.drawPath(
      cornerPath,
      highContrast ? _headHighContrastCornerPaint : _headCornerPaint,
    );
  }

  static void _paintTorsoContour(
    Canvas canvas, {
    required Size size,
    required List<Offset> posePoints,
    required double shoulderWidth,
    required bool highContrast,
  }) {
    if (posePoints.length < 3 || shoulderWidth < 8.0) return;
    final ls = posePoints[1];
    final rs = posePoints[2];
    final shoulderSpanVec = rs - ls;
    final rightDir = _norm(shoulderSpanVec, const Offset(1, 0));
    const downDir = Offset(0, 1);
    final torsoMid = (ls + rs) * 0.5;
    final spineDown = _norm(
      Offset(-rightDir.dy, rightDir.dx).dy >= 0
          ? Offset(-rightDir.dy, rightDir.dx)
          : Offset(rightDir.dy, -rightDir.dx),
      downDir,
    );

    final Offset lHip;
    final Offset rHip;
    if (posePoints.length >= 11) {
      final rawLHip = posePoints[9];
      final rawRHip = posePoints[10];
      final validHips = rawLHip.dy > ls.dy + shoulderWidth * 0.35 &&
          rawRHip.dy > rs.dy + shoulderWidth * 0.35 &&
          rawLHip.dy <= size.height * 1.45 &&
          rawRHip.dy <= size.height * 1.45;
      if (validHips) {
        lHip = Offset(
          rawLHip.dx.clamp(0.0, size.width),
          rawLHip.dy.clamp(ls.dy + shoulderWidth * 0.55, size.height + 24),
        );
        rHip = Offset(
          rawRHip.dx.clamp(0.0, size.width),
          rawRHip.dy.clamp(rs.dy + shoulderWidth * 0.55, size.height + 24),
        );
      } else {
        final estDrop = shoulderWidth * 1.15;
        lHip = torsoMid +
            spineDown * estDrop -
            rightDir * (shoulderWidth * 0.36);
        rHip = torsoMid +
            spineDown * estDrop +
            rightDir * (shoulderWidth * 0.36);
      }
    } else {
      final estDrop = shoulderWidth * 1.15;
      lHip = torsoMid +
          spineDown * estDrop -
          rightDir * (shoulderWidth * 0.36);
      rHip = torsoMid +
          spineDown * estDrop +
          rightDir * (shoulderWidth * 0.36);
    }

    final neckDip = torsoMid + spineDown * (shoulderWidth * 0.08);
    final outerRadius = (shoulderWidth * 0.065).clamp(4.0, 14.0);
    final lOuter = ls - rightDir * outerRadius + spineDown * (outerRadius * 0.6);
    final rOuter = rs + rightDir * outerRadius + spineDown * (outerRadius * 0.6);
    final lRib = Offset.lerp(lOuter, lHip, 0.5)! + rightDir * (outerRadius * 0.35);
    final rRib = Offset.lerp(rOuter, rHip, 0.5)! - rightDir * (outerRadius * 0.35);
    final waistMid = (lHip + rHip) * 0.5 + spineDown * (outerRadius * 0.4);

    final torsoPath = Path()
      ..moveTo(ls.dx, ls.dy)
      ..quadraticBezierTo(neckDip.dx, neckDip.dy, rs.dx, rs.dy)
      ..quadraticBezierTo(rOuter.dx, rOuter.dy, rRib.dx, rRib.dy)
      ..quadraticBezierTo(rHip.dx, rHip.dy, waistMid.dx, waistMid.dy)
      ..quadraticBezierTo(lHip.dx, lHip.dy, lRib.dx, lRib.dy)
      ..quadraticBezierTo(lOuter.dx, lOuter.dy, ls.dx, ls.dy);

    canvas.drawPath(
      torsoPath,
      highContrast ? _torsoHighContrastPaint : _torsoContourPaint,
    );
  }

  static void _paintArmContours(
    Canvas canvas, {
    required List<Offset> posePoints,
    required List<List<Offset>> hands,
    required double shoulderWidth,
    required bool highContrast,
  }) {
    final baseHalfW = (shoulderWidth * 0.085).clamp(5.5, 13.5);
    final elbowHalfW = baseHalfW * 0.85;
    final wristHalfW = baseHalfW * 0.68;

    for (final chain in const [(1, 3, 5), (2, 4, 6)]) {
      final sIdx = chain.$1;
      final eIdx = chain.$2;
      final wIdx = chain.$3;
      if (wIdx >= posePoints.length) continue;

      final s = posePoints[sIdx];
      final e = posePoints[eIdx];
      var w = posePoints[wIdx];
      var hasAttachedHand = false;
      var bestDist = (shoulderWidth * 0.42).clamp(24.0, 90.0);
      for (final hand in hands) {
        if (hand.isEmpty) continue;
        final d = (hand.first - w).distance;
        if (d <= bestDist) {
          bestDist = d;
          w = hand.first;
          hasAttachedHand = true;
        }
      }

      final upperLen = (e - s).distance;
      final foreLen = (w - e).distance;
      if (upperLen < 4.0 && foreLen < 4.0) continue;

      final uDir = _norm(e - s, const Offset(0, 1));
      final fDir = _norm(w - e, uDir);
      final uPerp = Offset(-uDir.dy, uDir.dx);
      final fPerp = Offset(-fDir.dy, fDir.dx);
      final ePerp = _norm(uPerp + fPerp, fPerp);

      final sA = s + uPerp * baseHalfW;
      final sB = s - uPerp * baseHalfW;
      final eA = e + ePerp * elbowHalfW;
      final eB = e - ePerp * elbowHalfW;
      // Stop slightly before the wrist collar when a hand contour is active so
      // arm edges meet the wrist cuff seamlessly without crossing inside palm.
      final wEnd = hasAttachedHand && foreLen > 12.0
          ? w - fDir * (wristHalfW * 0.45)
          : w;
      final wA = wEnd + fPerp * wristHalfW;
      final wB = wEnd - fPerp * wristHalfW;

      final armPath = Path()
        ..moveTo(sA.dx, sA.dy)
        ..lineTo(eA.dx, eA.dy)
        ..lineTo(wA.dx, wA.dy)
        ..moveTo(sB.dx, sB.dy)
        ..lineTo(eB.dx, eB.dy)
        ..lineTo(wB.dx, wB.dy);

      if (!highContrast) {
        canvas.drawPath(armPath, _armOuterPaint);
      }
      canvas.drawPath(
        armPath,
        highContrast ? _armHighContrastPaint : _armLinePaint,
      );

      // Minimal elbow tracking indicator.
      _drawTrackingIndicator(
        canvas,
        e,
        ringRadius: 3.8,
        coreRadius: 1.5,
        highContrast: highContrast,
      );

      // If hand landmarks aren't present on this wrist, show a minimal wrist ring.
      if (!hasAttachedHand) {
        _drawTrackingIndicator(
          canvas,
          w,
          ringRadius: 4.0,
          coreRadius: 1.7,
          highContrast: highContrast,
        );
      }
    }
  }

  /// Constructs an organic hand outline path and wrist collar arc matching the
  /// luminous cyan/teal contour aesthetic from 21 MediaPipe hand landmarks.
  static ({Path outline, Path collar, Offset wristCenter}) buildHandContourPaths(
    List<Offset> points,
  ) {
    if (points.length < 21) {
      final path = Path();
      if (points.isNotEmpty) {
        path.moveTo(points.first.dx, points.first.dy);
        for (var i = 1; i < points.length; i++) {
          path.lineTo(points[i].dx, points[i].dy);
        }
      }
      return (
        outline: path,
        collar: Path(),
        wristCenter: points.isNotEmpty ? points.first : Offset.zero,
      );
    }

    final wrist = points[0];
    final midMcp = points[9];
    final indexMcp = points[5];
    final pinkyMcp = points[17];
    final palmAxis = _norm(midMcp - wrist, const Offset(0, -1));
    final palmPx = math.max(12.0, (midMcp - wrist).distance);
    final palmCenter = (wrist + indexMcp + midMcp + pinkyMcp) * 0.25;

    final v5 = indexMcp - wrist;
    final v17 = pinkyMcp - wrist;
    final cross = v5.dx * v17.dy - v5.dy * v17.dx;
    final sideSign = cross >= 0 ? 1.0 : -1.0;
    final palmPinkyPerp = Offset(-palmAxis.dy, palmAxis.dx) * sideSign;

    final d1 = (points[9] - points[5]).distance;
    final d2 = (points[13] - points[9]).distance;
    final d3 = (points[17] - points[13]).distance;
    final mcpSpacing = (d1 + d2 + d3) / 3.0;
    final fingerHalfW =
        (mcpSpacing * 0.34).clamp(palmPx * 0.072, palmPx * 0.128);

    final wristThumbCorner =
        wrist - palmPinkyPerp * (palmPx * 0.30) + palmAxis * (palmPx * 0.04);
    final wristPinkyCorner =
        wrist + palmPinkyPerp * (palmPx * 0.28) + palmAxis * (palmPx * 0.04);
    final wristMidCollar = wrist + palmAxis * (palmPx * 0.015);

    final contourPts = <Offset>[wristThumbCorner];

    Offset segPinkyPerp(int iA, int iB, Offset fallback) {
      final dir = _norm(points[iB] - points[iA], fallback);
      return Offset(-dir.dy, dir.dx) * sideSign;
    }

    void appendFinger(List<int> idx, double scaleMul, {bool isThumb = false}) {
      final rBase = fingerHalfW * scaleMul;
      final rMid = rBase * 0.84;
      final rTip = rBase * 0.72;
      final mcp = points[idx[0]];
      final pip = points[idx[1]];
      final dip = points[idx[2]];
      final tip = points[idx[3]];

      if (!isThumb) {
        // Check if finger is tightly curled downward into the palm so we trace
        // the dorsal knuckle crest cleanly rather than folding inside the palm.
        final tipToPalm = (tip - palmCenter).distance;
        final pipToPalm = (pip - palmCenter).distance;
        final tipToWrist = (tip - wrist).distance;
        final pipToWrist = (pip - wrist).distance;
        final isCurled =
            tipToPalm < pipToPalm * 0.98 && tipToWrist < pipToWrist * 0.96;
        if (isCurled) {
          final kDir = _norm(pip - mcp, palmAxis);
          final kPerp = Offset(-kDir.dy, kDir.dx) * sideSign;
          contourPts
            ..add(mcp - kPerp * rBase)
            ..add(pip - kPerp * rMid + kDir * (rMid * 0.25))
            ..add(pip + kDir * (rMid * 0.85))
            ..add(pip + kPerp * rMid + kDir * (rMid * 0.25))
            ..add(mcp + kPerp * rBase);
          return;
        }
      }

      final dir0 = _norm(pip - mcp, palmAxis);
      final dir1 = _norm(dip - pip, dir0);
      final dir2 = _norm(tip - dip, dir1);

      final p0 = segPinkyPerp(idx[0], idx[1], palmPinkyPerp);
      final p1 = segPinkyPerp(idx[1], idx[2], p0);
      final p2 = segPinkyPerp(idx[2], idx[3], p1);
      final pBase = _norm(p0, palmPinkyPerp);
      final pMid1 = _norm(p0 + p1, p0);
      final pMid2 = _norm(p1 + p2, p2);

      // Walk up thumb side -> rounded tip dome -> down pinky side.
      contourPts
        ..add(mcp - pBase * rBase + dir0 * (palmPx * 0.06))
        ..add(pip - pMid1 * rMid)
        ..add(dip - pMid2 * (rTip * 1.05))
        ..add(tip - p2 * rTip)
        ..add(tip + dir2 * (rTip * 1.12))
        ..add(tip + p2 * rTip)
        ..add(dip + pMid2 * (rTip * 1.05))
        ..add(pip + pMid1 * rMid);
      if (!isThumb) {
        contourPts.add(mcp + pBase * rBase + dir0 * (palmPx * 0.06));
      }
    }

    // Thumb outer side from wrist corner -> CMC -> MCP -> IP -> tip -> inner IP.
    appendFinger(const [1, 2, 3, 4], 1.12, isThumb: true);

    // Bridge thumb-index web smoothly.
    final thumbWeb = Offset.lerp(points[2], points[5], 0.5)! -
        palmPinkyPerp * (palmPx * 0.05) +
        palmAxis * (palmPx * 0.05);
    contourPts.add(thumbWeb);

    // Index -> Middle -> Ring -> Pinky.
    appendFinger(const [5, 6, 7, 8], 1.0);
    appendFinger(const [9, 10, 11, 12], 1.02);
    appendFinger(const [13, 14, 15, 16], 0.96);
    appendFinger(const [17, 18, 19, 20], 0.86);

    // Outer hypothenar palm contour down to pinky wrist corner.
    final palmSideMid = Offset.lerp(pinkyMcp, wristPinkyCorner, 0.52)! +
        palmPinkyPerp * (fingerHalfW * 0.85);
    contourPts
      ..add(palmSideMid)
      ..add(wristPinkyCorner);

    final outline = _smoothOpenPolyline(contourPts);

    final collar = Path()
      ..moveTo(wristThumbCorner.dx, wristThumbCorner.dy)
      ..quadraticBezierTo(
        wristMidCollar.dx,
        wristMidCollar.dy,
        wristPinkyCorner.dx,
        wristPinkyCorner.dy,
      );

    return (
      outline: outline,
      collar: collar,
      wristCenter: wristMidCollar,
    );
  }

  static void _paintHandContour(
    Canvas canvas,
    List<Offset> points, {
    required bool highContrast,
  }) {
    if (points.isEmpty) return;
    final built = buildHandContourPaths(points);

    if (!highContrast) {
      canvas.drawPath(built.outline, _handHaloPaint);
      canvas.drawPath(built.collar, _armOuterPaint);
    }
    canvas.drawPath(
      built.outline,
      highContrast ? _handHighContrastPaint : _handCorePaint,
    );
    canvas.drawPath(
      built.collar,
      highContrast ? _armHighContrastPaint : _armLinePaint,
    );

    // Subtle crisp fingertip indicators.
    for (final tipIdx in handLandmarkFingertips) {
      if (tipIdx >= points.length) continue;
      final tip = points[tipIdx];
      canvas.drawCircle(tip, highContrast ? 4.8 : 4.2, _fingertipRingPaint);
      canvas.drawCircle(
        tip,
        highContrast ? 2.5 : 2.0,
        _indicatorCorePaint,
      );
    }

    // Minimal ringed wrist indicator centered on the wrist cuff.
    _drawTrackingIndicator(
      canvas,
      built.wristCenter,
      ringRadius: 4.6,
      coreRadius: 2.1,
      highContrast: highContrast,
    );
  }

  static void _drawTrackingIndicator(
    Canvas canvas,
    Offset center, {
    required double ringRadius,
    required double coreRadius,
    required bool highContrast,
  }) {
    canvas.drawCircle(center, ringRadius + 1.6, _jointRingHaloPaint);
    canvas.drawCircle(
      center,
      ringRadius,
      highContrast ? _armHighContrastPaint : _jointRingStrokePaint,
    );
    canvas.drawCircle(center, coreRadius, _indicatorCorePaint);
  }

  static Path _smoothOpenPolyline(List<Offset> pts) {
    final path = Path();
    if (pts.isEmpty) return path;
    path.moveTo(pts.first.dx, pts.first.dy);
    if (pts.length == 2) {
      path.lineTo(pts[1].dx, pts[1].dy);
      return path;
    }
    for (var i = 1; i < pts.length - 1; i++) {
      final mid = (pts[i] + pts[i + 1]) * 0.5;
      path.quadraticBezierTo(pts[i].dx, pts[i].dy, mid.dx, mid.dy);
    }
    path.lineTo(pts.last.dx, pts.last.dy);
    return path;
  }

  static Offset _norm(Offset v, Offset fallback) {
    final len = v.distance;
    return len > 1e-4 ? ui.Offset(v.dx / len, v.dy / len) : fallback;
  }
}
