import 'dart:math' as math;

import '../l10n/app_localizations.dart';
import 'alphabet_sign_capture.dart';
import 'landmark_service.dart';
import 'sign_capture.dart';
import 'sign_classifier.dart';

enum RecognizedTokenKind { letter, dynamicSign, space }

class RecognizedToken {
  const RecognizedToken._(this.kind, this.label);

  const RecognizedToken.letter(String label)
      : this._(RecognizedTokenKind.letter, label);

  const RecognizedToken.sign(String label)
      : this._(RecognizedTokenKind.dynamicSign, label);

  const RecognizedToken.space() : this._(RecognizedTokenKind.space, ' ');

  final RecognizedTokenKind kind;
  final String label;
}

/// Coordinates continuous dynamic FSL sign recognition and static FSL alphabet
/// fingerspelling within a single camera stream without restarting between signs.
///
/// Separates temporary candidate predictions from confirmed output tokens,
/// tracks multi-frame posture stability vs. movement trajectories, and gives
/// confirmed dynamic gestures precedence over overlapping static handshapes.
class ContinuousSignSession {
  ContinuousSignSession({
    required this.dynamicCapture,
    this.alphabetCapture,
  });

  final SignCapture dynamicCapture;
  final AlphabetSignCapture? alphabetCapture;

  final List<RecognizedToken> _tokens = <RecognizedToken>[];

  String? _pendingLetter;
  double _pendingScore = 0.0;
  SignPrediction? _pendingPrediction;
  int? _pendingStartMs;
  int _pendingSteadyFrames = 0;

  bool _suppressAlphabetUntilRelease = false;
  SignPrediction? _temporaryPrediction;
  SignPrediction? _confirmedBadge;

  int _staticCommitsInPresence = 0;
  int? _lastStaticCommitMs;
  int? _departureAfterStaticMs;
  int _consumedAlphaCount = 0;

  // Automatic word-boundary state triggered by intentionally lowering hand(s)
  // into a brief neutral resting position after fingerspelling a letter.
  double? _lastLetterWristY;
  double? _lastLetterPalm;
  double? _prevSpacingWristY;
  int? _prevSpacingSampleMs;
  double? _neutralRestWristY;
  bool _sawLoweredHandAfterLetter = false;
  bool _inNeutralRest = false;
  bool _restWasTwoHanded = false;
  bool _pendingWordBreak = false;
  int? _loweredRestStartMs;
  int? _twoHandRestStartMs;
  int? _lastHandSeenForSpacingMs;
  int? _lastSpacingTimestampMs;

  List<RecognizedToken> get tokens => List<RecognizedToken>.unmodifiable(_tokens);
  bool get hasTokens => _tokens.isNotEmpty;
  bool get canInsertSpace =>
      _tokens.isNotEmpty && _tokens.last.kind != RecognizedTokenKind.space;
  bool get isWordBreakPending => _pendingWordBreak;

  /// Live unconfirmed candidate prediction (never written directly to [tokens]).
  SignPrediction? get temporaryPrediction => _temporaryPrediction;

  /// Prediction surfaced in the UI status badge: shows a live candidate when
  /// present, or the most recently confirmed sign/letter otherwise.
  SignPrediction? get activeBadgePrediction =>
      _temporaryPrediction ?? _confirmedBadge;

  bool get _isDynamicMovementBlocking =>
      dynamicCapture.isDynamicGestureCandidate &&
      !dynamicCapture.isSettledInPlaceMorph;

  bool get retry =>
      (!dynamicCapture.hasActiveMovement &&
          !_isDynamicMovementBlocking &&
          !_suppressAlphabetUntilRelease &&
          _temporaryPrediction == null &&
          (alphabetCapture?.retry ?? false)) ||
      (dynamicCapture.retry &&
          dynamicCapture.result == null &&
          _pendingLetter == null &&
          _temporaryPrediction == null &&
          (alphabetCapture?.activePrediction == null));

  String formatText(AppLocalizations l10n) {
    if (_tokens.isEmpty) return '';
    final buffer = StringBuffer();
    RecognizedTokenKind? previousKind;
    for (final token in _tokens) {
      switch (token.kind) {
        case RecognizedTokenKind.letter:
          if (previousKind == RecognizedTokenKind.dynamicSign) {
            buffer.write(' ');
          }
          buffer.write(token.label);
        case RecognizedTokenKind.dynamicSign:
          if (previousKind != null &&
              previousKind != RecognizedTokenKind.space) {
            buffer.write(' ');
          }
          buffer.write(l10n.signLabel(token.label));
        case RecognizedTokenKind.space:
          if (previousKind != null &&
              previousKind != RecognizedTokenKind.space) {
            buffer.write(' ');
          }
      }
      previousKind = token.kind;
    }
    return buffer.toString();
  }

  void addFrame(LandmarkFrame frame, int timestampMs, {required bool useBody}) {
    final dynamicFrame = useBody ? frame.modelBody : frame.modelHands;
    final prevDynamicCount = dynamicCapture.labels.length;
    final prevAlphaCount = alphabetCapture?.letters.length ?? 0;
    final wasWaitingForRelease = dynamicCapture.isWaitingForRelease;

    if (_lastSpacingTimestampMs != null &&
        timestampMs < _lastSpacingTimestampMs!) {
      _resetWordSpacingTracking();
    }
    _lastSpacingTimestampMs = timestampMs;

    dynamicCapture.add(dynamicFrame, timestampMs);

    if (dynamicFrame.isEmpty) {
      _updateWordSpacingOnEmpty(timestampMs);
      alphabetCapture?.add(const [], timestampMs);
      if (dynamicCapture.labels.length > prevDynamicCount) {
        if (dynamicCapture.lastFinishWasDynamic) {
          _commitDynamicSigns(
            prevDynamicCount,
            timestampMs: timestampMs,
            hasHands: false,
          );
          return;
        }
        dynamicCapture.labels.removeRange(
          prevDynamicCount,
          dynamicCapture.labels.length,
        );
        dynamicCapture.result = null;
        dynamicCapture.retry = false;
      }
      if (!dynamicCapture.recording) {
        _clearPendingLetter();
        _temporaryPrediction = null;
      }
      if (!dynamicCapture.isWaitingForRelease) {
        _suppressAlphabetUntilRelease = false;
        _resetPresenceHistory();
        _consumedAlphaCount = alphabetCapture?.letters.length ?? 0;
      }
      if ((alphabetCapture?.activePrediction == null) &&
          !dynamicCapture.recording &&
          !dynamicCapture.isWaitingForRelease) {
        _confirmedBadge = null;
      }
      return;
    }

    if (wasWaitingForRelease &&
        !dynamicCapture.isWaitingForRelease &&
        dynamicCapture.recording) {
      _departureAfterStaticMs ??= timestampMs;
    }

    if (dynamicCapture.labels.length > prevDynamicCount) {
      if (dynamicCapture.lastFinishWasDynamic) {
        _commitDynamicSigns(
          prevDynamicCount,
          timestampMs: timestampMs,
          hasHands: true,
        );
        return;
      }
      dynamicCapture.labels.removeRange(
        prevDynamicCount,
        dynamicCapture.labels.length,
      );
      dynamicCapture.result = null;
      dynamicCapture.retry = false;
    }

    if (_suppressAlphabetUntilRelease) {
      if (dynamicCapture.isWaitingForRelease) {
        alphabetCapture?.reset();
        _consumedAlphaCount = 0;
        _clearPendingLetter();
        _temporaryPrediction = null;
        return;
      }
      _suppressAlphabetUntilRelease = false;
    }

    final alpha = alphabetCapture;
    if (alpha == null) return;

    final alphaHands = frame.modelHands3D;
    _updateWordSpacingOnHandsPresent(
      alphaHands,
      timestampMs,
      pose: frame.pose,
    );

    if (_inNeutralRest) {
      alpha.reset();
      _consumedAlphaCount = 0;
      _clearPendingLetter();
      _temporaryPrediction = null;
      return;
    }

    if (alphaHands.length != 1 || dynamicCapture.sawTwoHands) {
      _clearPendingLetter();
      _temporaryPrediction = null;
      alpha.reset();
      _consumedAlphaCount = 0;
      if (!dynamicCapture.recording && !dynamicCapture.isWaitingForRelease) {
        _confirmedBadge = null;
      }
      return;
    }

    alpha.add(alphaHands, timestampMs);
    if (_consumedAlphaCount > alpha.letters.length) {
      _consumedAlphaCount = alpha.letters.length;
    }

    // Dynamic J and Z finger trajectories are confirmed by stroke geometry while
    // keeping gross wrist/arm travel small.
    if (alpha.letters.length > prevAlphaCount && alpha.isDynamicLetterActive) {
      _consumedAlphaCount = alpha.letters.length;
      if (!dynamicCapture.sawTwoHands && dynamicCapture.maxTravel <= 1.25) {
        final letter = alpha.letters.last;
        final pred = alpha.activePrediction;
        final score = pred?.score ?? alpha.classifier.threshold;
        _confirmStaticLetter(
          letter,
          pred,
          score,
          dynamicFrame,
          timestampMs,
        );
      } else {
        _clearPendingLetter();
        _temporaryPrediction = null;
      }
      return;
    }

    // Active dynamic movement suppresses static letter staging/confirmation
    // so mid-gesture handshapes never leak into confirmed text.
    final dynamicBlocking = _isDynamicMovementBlocking;
    if (dynamicBlocking || dynamicCapture.hasActiveMovement) {
      if (dynamicBlocking && dynamicCapture.wristMaxTravel >= 0.45) {
        _consumedAlphaCount = alpha.letters.length;
      }
      _clearPendingLetter();
      _temporaryPrediction = null;
      return;
    }

    if (alpha.letters.length > _consumedAlphaCount) {
      _consumedAlphaCount = alpha.letters.length;
      final letter = alpha.letters.last;
      final pred = alpha.activePrediction;
      if (pred != null && pred.label == letter) {
        final score = pred.score;
        if (_staticCommitsInPresence >= 1 &&
            alpha.isPostureSteady &&
            score >= 0.78 &&
            alpha.lastStepDelta < 0.16 &&
            !dynamicCapture.hasActiveMovement &&
            !dynamicBlocking) {
          _confirmStaticLetter(
            letter,
            pred,
            score,
            dynamicFrame,
            timestampMs,
          );
          return;
        }
        _pendingLetter = letter;
        _pendingScore = score;
        _pendingPrediction = pred;
        _pendingStartMs = timestampMs;
        _pendingSteadyFrames = alpha.isPostureSteady ? 1 : 0;
        _temporaryPrediction = pred;
        return;
      }
    }

    if (_pendingLetter != null) {
      final current = alpha.activePrediction;
      if (current == null ||
          current.label != _pendingLetter ||
          !alpha.isPostureSteady) {
        _clearPendingLetter();
        _temporaryPrediction = current;
        return;
      }
      _pendingSteadyFrames++;
      _pendingPrediction = current;
      _pendingScore = current.score;
      _temporaryPrediction = current;
      final holdAfterLockMs = timestampMs - (_pendingStartMs ?? timestampMs);
      final fastFollowup = _staticCommitsInPresence >= 1 &&
          alpha.isPostureSteady &&
          _pendingScore >= 0.72;
      final minSteadyFrames = fastFollowup ? 1 : 2;
      final minHoldAfterLock = fastFollowup ? 0 : 80;
      if (_pendingSteadyFrames >= minSteadyFrames &&
          holdAfterLockMs >= minHoldAfterLock &&
          !dynamicCapture.hasActiveMovement &&
          !dynamicBlocking) {
        _confirmStaticLetter(
          _pendingLetter!,
          _pendingPrediction,
          _pendingScore,
          dynamicFrame,
          timestampMs,
        );
      }
      return;
    }

    _temporaryPrediction = alpha.activePrediction;
  }

  void _confirmStaticLetter(
    String letter,
    SignPrediction? prediction,
    double score,
    SignFrame dynamicFrame,
    int timestampMs,
  ) {
    if (_pendingWordBreak && canInsertSpace) {
      _tokens.add(const RecognizedToken.space());
    }
    _resetWordSpacingTracking();

    final threshold = alphabetCapture?.classifier.threshold ?? 0.0;
    final badge = prediction ?? SignPrediction(letter, score, threshold);
    _tokens.add(RecognizedToken.letter(letter));
    _confirmedBadge = badge;
    _temporaryPrediction = null;
    _clearPendingLetter();
    _staticCommitsInPresence++;
    _lastStaticCommitMs = timestampMs;
    _departureAfterStaticMs = null;
    if (dynamicFrame.isNotEmpty) {
      final wrist = dynamicFrame.first.first;
      final mid = dynamicFrame.first[9];
      _lastLetterWristY = wrist.y;
      _lastLetterPalm = math.max(0.04, _dist2d(mid.x - wrist.x, mid.y - wrist.y));
      dynamicCapture.latchStaticPosture(dynamicFrame, timestampMs);
    }
  }

  void _commitDynamicSigns(
    int prevCount, {
    required int timestampMs,
    required bool hasHands,
  }) {
    _clearPendingLetter();
    _temporaryPrediction = null;
    // If a single brief static letter was committed from the starting pose of
    // this same gesture right before sweeping into the dynamic stroke, replace
    // that preparation artifact with the confirmed dynamic gesture.
    if (_staticCommitsInPresence == 1 &&
        _lastStaticCommitMs != null &&
        _tokens.isNotEmpty &&
        _tokens.last.kind == RecognizedTokenKind.letter) {
      final departureGap =
          (_departureAfterStaticMs ?? timestampMs) - _lastStaticCommitMs!;
      final totalElapsed = timestampMs - _lastStaticCommitMs!;
      if (departureGap <= 650 && totalElapsed <= 3600) {
        _tokens.removeLast();
      }
    }
    _resetPresenceHistory();
    _resetWordSpacingTracking();

    for (var i = prevCount; i < dynamicCapture.labels.length; i++) {
      _tokens.add(RecognizedToken.sign(dynamicCapture.labels[i]));
    }
    _confirmedBadge = dynamicCapture.result;
    alphabetCapture?.reset();
    _consumedAlphaCount = 0;
    _suppressAlphabetUntilRelease =
        hasHands && dynamicCapture.isWaitingForRelease;
  }

  void _updateWordSpacingOnEmpty(int timestampMs) {
    if (_tokens.isEmpty || _tokens.last.kind != RecognizedTokenKind.letter) {
      _resetWordSpacingTracking();
      return;
    }
    final lastSeen = _lastHandSeenForSpacingMs;
    if (_sawLoweredHandAfterLetter &&
        lastSeen != null &&
        timestampMs - lastSeen >= 180) {
      _pendingWordBreak = true;
      _inNeutralRest = false;
      _loweredRestStartMs = null;
      _twoHandRestStartMs = null;
      _restWasTwoHanded = false;
    }
  }

  void _updateWordSpacingOnHandsPresent(
    List<List<LandmarkPoint>> alphaHands,
    int timestampMs, {
    required List<LandmarkPoint> pose,
  }) {
    final prevSeen = _lastHandSeenForSpacingMs;
    _lastHandSeenForSpacingMs = timestampMs;

    if (_tokens.isEmpty || _tokens.last.kind != RecognizedTokenKind.letter) {
      _resetWordSpacingTracking();
      return;
    }
    if (alphaHands.isEmpty) return;

    if (_sawLoweredHandAfterLetter &&
        prevSeen != null &&
        timestampMs - prevSeen >= 180) {
      _pendingWordBreak = true;
    }

    var highestWristY = double.infinity;
    var palmSum = 0.0;
    var palmCount = 0;
    for (final hand in alphaHands) {
      if (hand.isEmpty) continue;
      final wy = hand.first.y;
      if (wy < highestWristY) highestWristY = wy;
      if (hand.length > 9) {
        palmSum += math.max(
          0.04,
          _dist2d(hand[9].x - hand[0].x, hand[9].y - hand[0].y),
        );
        palmCount++;
      }
    }
    if (!highestWristY.isFinite) return;

    final meanPalm = palmCount > 0 ? palmSum / palmCount : 0.12;
    final refPalm = math.max(0.05, _lastLetterPalm ?? meanPalm);
    final refWristY = _lastLetterWristY ?? highestWristY;
    final dropFromLetter = highestWristY - refWristY;

    var poseBelowSigningZone = false;
    if (pose.length >= 5) {
      final shoulderMidY = (pose[1].y + pose[2].y) * 0.5;
      final shoulderSpan = math.max(
        0.08,
        _dist2d(pose[2].x - pose[1].x, pose[2].y - pose[1].y),
      );
      if (highestWristY >= shoulderMidY + 0.72 * shoulderSpan &&
          dropFromLetter >= 0.45 * refPalm) {
        poseBelowSigningZone = true;
      }
    }

    final strongDownDrop =
        dropFromLetter >= math.max(0.75 * refPalm, 0.070);
    final nearBottomRest =
        highestWristY >= 0.65 && dropFromLetter >= 0.45 * refPalm;
    final isLowered =
        strongDownDrop || nearBottomRest || poseBelowSigningZone;
    final deepSingleRest =
        highestWristY >= 0.72 &&
        dropFromLetter >= math.max(1.2 * refPalm, 0.14);
    final inFrameRestEligible =
        poseBelowSigningZone ||
        deepSingleRest ||
        (alphaHands.length >= 2 && isLowered && highestWristY >= 0.55);

    if (isLowered) {
      _sawLoweredHandAfterLetter = true;
      final stepY = _prevSpacingWristY != null
          ? (highestWristY - _prevSpacingWristY!).abs()
          : 0.0;
      final maxRestStep =
          (alphaHands.length >= 2 ? 0.16 : 0.14) * refPalm;
      if (inFrameRestEligible && stepY <= maxRestStep) {
        final calmSinceMs = _prevSpacingSampleMs ?? timestampMs;
        _loweredRestStartMs ??= calmSinceMs;
        if (alphaHands.length >= 2) {
          _twoHandRestStartMs ??= calmSinceMs;
        }
        final restElapsed = timestampMs - _loweredRestStartMs!;
        final twoHandElapsed = _twoHandRestStartMs != null
            ? timestampMs - _twoHandRestStartMs!
            : 0;
        if (restElapsed >= 140 || twoHandElapsed >= 140) {
          _inNeutralRest = true;
          _pendingWordBreak = true;
          _restWasTwoHanded = _restWasTwoHanded || alphaHands.length >= 2;
          _neutralRestWristY = _neutralRestWristY == null
              ? highestWristY
              : math.min(_neutralRestWristY!, highestWristY);
        }
      } else {
        _loweredRestStartMs = null;
        _twoHandRestStartMs = null;
      }
    } else {
      if (_inNeutralRest) {
        final roseFromRest = highestWristY <=
                (_neutralRestWristY ?? refWristY) -
                    math.max(0.38 * refPalm, 0.06) ||
            dropFromLetter < 0.35 * refPalm;
        if (roseFromRest) {
          _inNeutralRest = false;
          _sawLoweredHandAfterLetter = false;
          _loweredRestStartMs = null;
          _twoHandRestStartMs = null;
          _restWasTwoHanded = false;
          _neutralRestWristY = null;
          dynamicCapture.resetSegment();
          alphabetCapture?.reset();
          _consumedAlphaCount = 0;
          _clearPendingLetter();
        }
      } else {
        _sawLoweredHandAfterLetter = false;
        _loweredRestStartMs = null;
        _twoHandRestStartMs = null;
        _restWasTwoHanded = false;
      }
    }

    _prevSpacingWristY = highestWristY;
    _prevSpacingSampleMs = timestampMs;
  }

  static double _dist2d(double dx, double dy) => math.sqrt(dx * dx + dy * dy);

  void _resetWordSpacingTracking({bool clearPendingBreak = true}) {
    if (clearPendingBreak) {
      _pendingWordBreak = false;
    }
    _sawLoweredHandAfterLetter = false;
    _inNeutralRest = false;
    _restWasTwoHanded = false;
    _loweredRestStartMs = null;
    _twoHandRestStartMs = null;
    _prevSpacingWristY = null;
    _prevSpacingSampleMs = null;
    _neutralRestWristY = null;
  }

  void _clearPendingLetter() {
    _pendingLetter = null;
    _pendingScore = 0.0;
    _pendingPrediction = null;
    _pendingStartMs = null;
    _pendingSteadyFrames = 0;
  }

  void _resetPresenceHistory() {
    _staticCommitsInPresence = 0;
    _lastStaticCommitMs = null;
    _departureAfterStaticMs = null;
  }

  void undo() {
    if (_tokens.isNotEmpty) {
      _tokens.removeLast();
    }
    _clearPendingLetter();
    _resetPresenceHistory();
    _resetWordSpacingTracking();
    _lastLetterWristY = null;
    _lastLetterPalm = null;
    _temporaryPrediction = null;
    _confirmedBadge = null;
    alphabetCapture?.reset();
    _consumedAlphaCount = 0;
  }

  void insertSpace() {
    if (!canInsertSpace) return;
    _tokens.add(const RecognizedToken.space());
    _clearPendingLetter();
    _resetWordSpacingTracking();
    _temporaryPrediction = null;
    alphabetCapture?.reset();
    _consumedAlphaCount = 0;
  }

  void clear() {
    _tokens.clear();
    _clearPendingLetter();
    _resetPresenceHistory();
    _resetWordSpacingTracking();
    _lastLetterWristY = null;
    _lastLetterPalm = null;
    _lastHandSeenForSpacingMs = null;
    _lastSpacingTimestampMs = null;
    _consumedAlphaCount = 0;
    _suppressAlphabetUntilRelease = false;
    _temporaryPrediction = null;
    _confirmedBadge = null;
    dynamicCapture.reset();
    alphabetCapture?.clear();
  }

  void resetSegment() {
    _clearPendingLetter();
    _resetPresenceHistory();
    _resetWordSpacingTracking();
    _lastLetterWristY = null;
    _lastLetterPalm = null;
    _lastHandSeenForSpacingMs = null;
    _consumedAlphaCount = 0;
    _suppressAlphabetUntilRelease = false;
    _temporaryPrediction = null;
    _confirmedBadge = null;
    dynamicCapture.resetSegment();
    alphabetCapture?.reset();
  }
}
