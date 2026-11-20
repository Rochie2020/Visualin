import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/alphabet_classifier.dart';
import '../services/alphabet_sign_capture.dart';
import '../services/continuous_sign_session.dart';
import '../services/sign_capture.dart';
import '../services/sign_classifier.dart';

import '../l10n/app_localizations.dart';
import '../services/app_preferences.dart';
import '../services/landmark_service.dart';
import '../theme/visualin_tokens.dart';
import '../widgets/app_scaffold.dart';
import '../widgets/camera_overlay_transform.dart';
import '../widgets/landmark_contour_painter.dart';
import '../widgets/landmark_overlay.dart';
import '../widgets/signing_workspace.dart';

typedef CamerasLoader = Future<List<CameraDescription>> Function();
typedef CameraControllerFactory = CameraController Function(
  CameraDescription description,
);
typedef SessionLoader = Future<ContinuousSignSession> Function();

/// Computes cover-fit dimensions matching [rawAspectRatio] without stretching
/// or leaving letterbox/pillarbox bars inside [viewportSize].
@visibleForTesting
Size computeCameraCoverSize({
  required Size viewportSize,
  required double rawAspectRatio,
  required DeviceOrientation orientation,
}) {
  if (!viewportSize.width.isFinite ||
      !viewportSize.height.isFinite ||
      viewportSize.width <= 0 ||
      viewportSize.height <= 0) {
    return Size.zero;
  }
  final isLandscape = orientation == DeviceOrientation.landscapeLeft ||
      orientation == DeviceOrientation.landscapeRight;
  final previewAspect = rawAspectRatio <= 0
      ? 1.0
      : (isLandscape ? rawAspectRatio : 1.0 / rawAspectRatio);
  final viewportAspect = viewportSize.width / viewportSize.height;
  if (viewportAspect > previewAspect) {
    return Size(viewportSize.width, viewportSize.width / previewAspect);
  }
  return Size(viewportSize.height * previewAspect, viewportSize.height);
}

class CameraTranslationScreen extends StatefulWidget {
  const CameraTranslationScreen({
    super.key,
    required this.landmarkService,
    this.preferences,
    @visibleForTesting this.camerasLoader,
    @visibleForTesting this.controllerFactory,
    @visibleForTesting this.sessionLoader,
    @visibleForTesting this.overlayDuration = const Duration(milliseconds: 1800),
    @visibleForTesting this.frameNow,
  });

  final LandmarkDetector landmarkService;
  final AppPreferences? preferences;
  final CamerasLoader? camerasLoader;
  final CameraControllerFactory? controllerFactory;
  final SessionLoader? sessionLoader;
  final Duration overlayDuration;
  final int Function()? frameNow;

  @override
  State<CameraTranslationScreen> createState() =>
      _CameraTranslationScreenState();
}

class _CameraTranslationScreenState extends State<CameraTranslationScreen> {
  CameraController? _controller;
  List<CameraDescription> _cameras = const <CameraDescription>[];
  final _overlay = LandmarkOverlay();
  final _hudNotifier = _HudStateNotifier();
  final _fullscreenOverlayToken = ValueNotifier<RecognizedToken?>(null);
  Timer? _overlayHideTimer;
  StreamSubscription<LandmarkFrame>? _landmarkSubscription;
  final _clock = Stopwatch()..start();
  ContinuousSignSession? _session;
  AlphabetClassifier? _alphabetModel;
  bool _bodyEnabled = false;
  bool _poseError = false;
  bool _isFullScreen = false;
  late TrackingOverlayStyle _overlayStyle =
      widget.preferences?.overlayStyle ?? TrackingOverlayStyle.contour;

  int? _rotation;
  bool _initializing = true;
  bool _isTearingDown = false;
  bool _isSwitchingCamera = false;

  @override
  void initState() {
    super.initState();
    _overlay.statusRevision.addListener(_notifyHudIfChanged);
    _setup();
  }

  void _notifyHudIfChanged() {
    final s = _session;
    final badge = s?.activeBadgePrediction;
    _hudNotifier.updateIfChanged(
      hasHands: _overlay.hands.isNotEmpty,
      hasPose: _overlay.pose.isNotEmpty,
      poseError: _poseError,
      tokenCount: s?.tokens.length ?? 0,
      lastTokenKind: s != null && s.tokens.isNotEmpty ? s.tokens.last.kind : null,
      lastTokenLabel: s != null && s.tokens.isNotEmpty ? s.tokens.last.label : '',
      badgeLabel: badge?.label,
      badgePercent: badge != null ? (badge.score * 100).round() : null,
      retry: s?.retry ?? false,
    );
  }

  void _toggleOverlayStyle() {
    final next = _overlayStyle == TrackingOverlayStyle.contour
        ? TrackingOverlayStyle.skeleton
        : TrackingOverlayStyle.contour;
    setState(() => _overlayStyle = next);
    unawaited(widget.preferences?.setOverlayStyle(next));
  }

  Future<void> _setup() async {
    CameraController? controller;
    try {
      if (widget.sessionLoader != null) {
        _session = await widget.sessionLoader!.call();
        if (_session!.dynamicCapture.classifier.features > 91) {
          _bodyEnabled = true;
          widget.landmarkService.setBodyTracking(true);
        }
      } else {
        final txModel = SignClassifier.decode(
          await rootBundle.loadString('assets/ml/transaction_signs.json'),
        );
        SignClassifier primaryModel = txModel;
        try {
          final combined = SignClassifier.decode(
            await rootBundle.loadString('assets/ml/fsl_combined_signs.json'),
          );
          primaryModel = combined.withFallback(txModel);
          _bodyEnabled = true;
          widget.landmarkService.setBodyTracking(true);
        } catch (_) {
          try {
            final fsl105 = SignClassifier.decode(
              await rootBundle.loadString('assets/ml/fsl105_signs.json'),
            );
            primaryModel = fsl105.withFallback(txModel);
            _bodyEnabled = true;
            widget.landmarkService.setBodyTracking(true);
          } catch (_) {
            // Non-fatal in test environments without bundled v4 models.
          }
        }
        final dynamicCapture = SignCapture(primaryModel);
        AlphabetSignCapture? alphabetCapture;
        try {
          _alphabetModel ??= await AlphabetClassifier.loadFromAssets();
          alphabetCapture = AlphabetSignCapture(_alphabetModel!);
        } catch (_) {
          // Non-fatal in widget tests or environments without bundled TFLite assets.
        }
        _session = ContinuousSignSession(
          dynamicCapture: dynamicCapture,
          alphabetCapture: alphabetCapture,
        );
      }
      final cameras = await (widget.camerasLoader ?? availableCameras)();
      if (!mounted || _isTearingDown) return;
      if (cameras.isEmpty) throw StateError('camera');
      _cameras = cameras;
      final selectedCamera = _preferredCamera(cameras);
      controller = widget.controllerFactory?.call(selectedCamera) ??
          CameraController(
            selectedCamera,
            ResolutionPreset.medium,
            imageFormatGroup: ImageFormatGroup.yuv420,
            enableAudio: false,
          );
      // Retain the controller before awaiting initialization so dispose can
      // tear it down if the user leaves this screen mid-initialization.
      _controller = controller;
      _landmarkSubscription =
          widget.landmarkService.frames.listen(_onLandmarkFrame);
      await controller.initialize();
      if (!mounted || _isTearingDown) return;
      await widget.landmarkService.start(
        controller,
        isActive: () => mounted && !_isTearingDown,
      );
      if (mounted && !_isTearingDown) {
        _notifyHudIfChanged();
        setState(() => _initializing = false);
      }
    } catch (_) {
      await _landmarkSubscription?.cancel();
      _landmarkSubscription = null;
      await widget.landmarkService.stop();
      if (!_isTearingDown && identical(_controller, controller)) {
        _controller = null;
        await controller?.dispose();
      }
      if (mounted && !_isTearingDown) {
        _notifyHudIfChanged();
        setState(() => _initializing = false);
      }
    }
  }

  void _onLandmarkFrame(LandmarkFrame frame) {
    // A queued stream event can arrive after cancel() has been requested.
    if (!mounted || _isTearingDown || _isSwitchingCamera) return;
    final session = _session;
    if (session == null) return;
    if (_rotation != null && frame.rotation != _rotation) {
      session.resetSegment();
      _overlay.reset();
    }
    _rotation = frame.rotation;
    final now = widget.frameNow?.call() ?? _clock.elapsedMilliseconds;
    final useBody =
        _bodyEnabled || session.dynamicCapture.classifier.features > 91;
    final prevCount = session.tokens.length;
    final prevLast = prevCount > 0 ? session.tokens.last : null;
    session.addFrame(frame, now, useBody: useBody);
    final nextCount = session.tokens.length;
    final nextLast = nextCount > 0 ? session.tokens.last : null;
    if (nextLast != null &&
        nextLast.kind != RecognizedTokenKind.space &&
        (nextCount != prevCount || !identical(nextLast, prevLast))) {
      _showConfirmedSignOverlay(nextLast);
    }
    if (frame.poseSampled && _poseError != frame.poseError) {
      setState(() => _poseError = frame.poseError);
    }
    _overlay.update(frame);
    _notifyHudIfChanged();
  }

  void _showConfirmedSignOverlay(RecognizedToken token) {
    if (!_isFullScreen) return;
    _overlayHideTimer?.cancel();
    _fullscreenOverlayToken.value = token;
    _overlayHideTimer = Timer(widget.overlayDuration, () {
      if (!mounted || _isTearingDown) return;
      _fullscreenOverlayToken.value = null;
    });
  }

  void _setFullScreen(bool value) {
    if (_isFullScreen == value) return;
    _overlayHideTimer?.cancel();
    _overlayHideTimer = null;
    _fullscreenOverlayToken.value = null;
    setState(() => _isFullScreen = value);
  }

  void _clearTranslation() {
    _overlayHideTimer?.cancel();
    _overlayHideTimer = null;
    _fullscreenOverlayToken.value = null;
    setState(() {
      _session?.clear();
      _notifyHudIfChanged();
    });
  }

  void _copyTranslation(AppLocalizations l10n, String text) {
    if (text.isEmpty) return;
    unawaited(Clipboard.setData(ClipboardData(text: text)));
    final messenger = ScaffoldMessenger.maybeOf(context);
    messenger?.hideCurrentSnackBar();
    messenger?.showSnackBar(
      SnackBar(
        content: Text(l10n.copiedToClipboard),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  CameraDescription _preferredCamera(List<CameraDescription> cameras) {
    return cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );
  }

  Future<void> _switchCamera() async {
    final previous = _controller;
    if (previous == null ||
        _cameras.length < 2 ||
        _isSwitchingCamera ||
        _isTearingDown) {
      return;
    }

    final currentIndex = _cameras.indexWhere(
      (camera) => camera == previous.description,
    );
    if (currentIndex < 0) return;
    final nextCamera = _cameras[(currentIndex + 1) % _cameras.length];
    setState(() {
      _isSwitchingCamera = true;
      _session?.resetSegment();
      _overlay.reset();
      _rotation = null;
    });

    try {
      await widget.landmarkService.stop();
      if (_isTearingDown) return;
      _controller = null;
      await previous.dispose();
      if (!mounted || _isTearingDown) return;

      final replacement = widget.controllerFactory?.call(nextCamera) ??
          CameraController(
            nextCamera,
            ResolutionPreset.medium,
            imageFormatGroup: ImageFormatGroup.yuv420,
            enableAudio: false,
          );
      _controller = replacement;
      await replacement.initialize();
      if (!mounted || _isTearingDown) return;
      await widget.landmarkService.start(
        replacement,
        isActive: () => mounted && !_isTearingDown,
      );
      if (mounted && !_isTearingDown) setState(() {});
    } catch (_) {
      if (mounted && !_isTearingDown) setState(() {});
    } finally {
      if (mounted && !_isTearingDown) {
        setState(() => _isSwitchingCamera = false);
      }
    }
  }

  @override
  void dispose() {
    _isTearingDown = true;
    _overlayHideTimer?.cancel();
    _overlayHideTimer = null;
    _fullscreenOverlayToken.dispose();
    _overlay.statusRevision.removeListener(_notifyHudIfChanged);
    _hudNotifier.dispose();
    final subscription = _landmarkSubscription;
    _landmarkSubscription = null;
    final controller = _controller;
    _controller = null;
    unawaited(_tearDownCamera(subscription, controller));
    widget.landmarkService.setBodyTracking(false);
    _overlay.dispose();
    super.dispose();
  }

  Future<void> _tearDownCamera(
    StreamSubscription<LandmarkFrame>? subscription,
    CameraController? controller,
  ) async {
    await subscription?.cancel();
    if (controller?.value.isStreamingImages == true) {
      await controller!.stopImageStream();
    }
    await widget.landmarkService.stop();
    await controller?.dispose();
  }

  String _formatTokenLabel(RecognizedToken token, AppLocalizations l10n) {
    if (token.kind == RecognizedTokenKind.space) return '';
    return l10n.signLabel(token.label);
  }

  String? _bodyHint(AppLocalizations l10n) {
    if (!_bodyEnabled) return null;
    if (_poseError) return l10n.bodyError;
    if (_overlay.pose.isEmpty) return l10n.bodyHint;
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return PopScope(
      canPop: !_isFullScreen,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop && _isFullScreen) _setFullScreen(false);
      },
      child: _isFullScreen
          ? _buildFullScreenScaffold(context, l10n)
          : _buildNormalScaffold(context, l10n),
    );
  }

  Widget _buildNormalScaffold(BuildContext context, AppLocalizations l10n) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 8, 24, 16),
          child: Column(
            children: [
              SizedBox(
                height: 48,
                child: Row(
                  children: [
                    IconButton(
                      constraints:
                          const BoxConstraints(minWidth: 44, minHeight: 44),
                      onPressed: () =>
                          Navigator.pushReplacementNamed(context, '/home'),
                      icon: const Icon(Icons.close),
                      tooltip: l10n.close,
                    ),
                    Expanded(
                      child: Text(
                        l10n.fslToText,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ),
                    Semantics(
                      button: true,
                      toggled: _overlayStyle == TrackingOverlayStyle.skeleton,
                      label: _overlayStyle == TrackingOverlayStyle.contour
                          ? l10n.switchToSkeletonOverlay
                          : l10n.switchToContourOverlay,
                      child: IconButton(
                        constraints:
                            const BoxConstraints(minWidth: 44, minHeight: 44),
                        onPressed: _toggleOverlayStyle,
                        icon: Icon(
                          _overlayStyle == TrackingOverlayStyle.skeleton
                              ? Icons.hub
                              : Icons.hub_outlined,
                          color: _overlayStyle == TrackingOverlayStyle.skeleton
                              ? VisualinColors.primaryOrange
                              : null,
                        ),
                        tooltip: _overlayStyle == TrackingOverlayStyle.contour
                            ? l10n.switchToSkeletonOverlay
                            : l10n.switchToContourOverlay,
                      ),
                    ),
                    IconButton(
                      constraints:
                          const BoxConstraints(minWidth: 44, minHeight: 44),
                      onPressed: _cameras.length > 1 && !_isSwitchingCamera
                          ? _switchCamera
                          : null,
                      icon: const Icon(Icons.flip_camera_ios_outlined),
                      tooltip: l10n.flipCamera,
                    ),
                  ],
                ),
              ),
              Expanded(
                child: SigningWorkspace(
                  preview: _CameraPreviewPanel(
                    overlay: _overlay,
                    controller: _controller,
                    initializing: _initializing,
                    cameraUnavailable: l10n.cameraUnavailable,
                    frameLabel: l10n.cameraFrameLabel,
                    overlayStyle: _overlayStyle,
                    highContrast: widget.preferences?.highContrast ??
                        MediaQuery.highContrastOf(context),
                    fullScreenLabel: l10n.fullScreen,
                    onEnterFullScreen: () => _setFullScreen(true),
                  ),
                  controls: [
                    _TranslationResultPanel(
                      overlay: _overlay,
                      hudListenable: _hudNotifier,
                      l10n: l10n,
                      session: () => _session,
                      bodyHint: () => _bodyHint(l10n),
                      onClear: _clearTranslation,
                      onCopy: (text) => _copyTranslation(l10n, text),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: const VisualinBottomNav(index: 1),
    );
  }

  Widget _buildFullScreenScaffold(
      BuildContext context, AppLocalizations l10n) {
    final reducedMotion = MediaQuery.disableAnimationsOf(context);
    return Scaffold(
      backgroundColor: Colors.black,
      body: AnnotatedRegion<SystemUiOverlayStyle>(
        value: SystemUiOverlayStyle.light,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Positioned.fill(
              child: _CameraPreviewPanel(
                overlay: _overlay,
                controller: _controller,
                initializing: _initializing,
                cameraUnavailable: l10n.cameraUnavailable,
                frameLabel: l10n.cameraFrameLabel,
                overlayStyle: _overlayStyle,
                highContrast: widget.preferences?.highContrast ??
                    MediaQuery.highContrastOf(context),
                isFullScreen: true,
              ),
            ),
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: SafeArea(
                bottom: false,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                  child: Row(
                    children: [
                      Flexible(
                        child: Semantics(
                          button: true,
                          label: l10n.exitFullScreen,
                          child: Material(
                            color: const Color(0xC21C1C1E),
                            shape: StadiumBorder(
                              side: BorderSide(
                                color: Colors.white.withValues(alpha: 0.24),
                              ),
                            ),
                            clipBehavior: Clip.antiAlias,
                            child: InkWell(
                              onTap: () => _setFullScreen(false),
                              child: ConstrainedBox(
                                constraints: const BoxConstraints(
                                  minHeight: 48,
                                  minWidth: 48,
                                ),
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 16,
                                    vertical: 10,
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      const Icon(
                                        Icons.fullscreen_exit,
                                        color: Colors.white,
                                        size: 20,
                                      ),
                                      const SizedBox(width: 8),
                                      Flexible(
                                        child: Text(
                                          l10n.exitFullScreen,
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis,
                                          style: Theme.of(context)
                                              .textTheme
                                              .labelLarge
                                              ?.copyWith(color: Colors.white),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                      const Spacer(),
                      Semantics(
                        button: true,
                        toggled: _overlayStyle == TrackingOverlayStyle.skeleton,
                        label: _overlayStyle == TrackingOverlayStyle.contour
                            ? l10n.switchToSkeletonOverlay
                            : l10n.switchToContourOverlay,
                        child: Material(
                          color: const Color(0xC21C1C1E),
                          shape: CircleBorder(
                            side: BorderSide(
                              color: Colors.white.withValues(alpha: 0.24),
                            ),
                          ),
                          clipBehavior: Clip.antiAlias,
                          child: IconButton(
                            constraints: const BoxConstraints(
                              minWidth: 48,
                              minHeight: 48,
                            ),
                            color: _overlayStyle == TrackingOverlayStyle.skeleton
                                ? VisualinColors.primaryOrange
                                : const Color(0xFF3BF0FF),
                            onPressed: _toggleOverlayStyle,
                            icon: Icon(
                              _overlayStyle == TrackingOverlayStyle.skeleton
                                  ? Icons.hub
                                  : Icons.hub_outlined,
                            ),
                            tooltip:
                                _overlayStyle == TrackingOverlayStyle.contour
                                    ? l10n.switchToSkeletonOverlay
                                    : l10n.switchToContourOverlay,
                          ),
                        ),
                      ),
                      if (_cameras.length > 1) ...[
                        const SizedBox(width: 12),
                        Material(
                          color: const Color(0xC21C1C1E),
                          shape: CircleBorder(
                            side: BorderSide(
                              color: Colors.white.withValues(alpha: 0.24),
                            ),
                          ),
                          clipBehavior: Clip.antiAlias,
                          child: IconButton(
                            constraints: const BoxConstraints(
                              minWidth: 48,
                              minHeight: 48,
                            ),
                            color: Colors.white,
                            onPressed:
                                !_isSwitchingCamera ? _switchCamera : null,
                            icon: const Icon(Icons.flip_camera_ios_outlined),
                            tooltip: l10n.flipCamera,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
            Positioned(
              left: 16,
              right: 16,
              bottom: 0,
              child: SafeArea(
                top: false,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      ValueListenableBuilder<RecognizedToken?>(
                        valueListenable: _fullscreenOverlayToken,
                        builder: (context, token, _) {
                          final label = token == null
                              ? ''
                              : _formatTokenLabel(token, l10n);
                          return AnimatedSwitcher(
                            duration: reducedMotion
                                ? Duration.zero
                                : const Duration(milliseconds: 180),
                            child: label.isEmpty
                                ? const SizedBox.shrink()
                                : Padding(
                                    key: ValueKey(
                                      'overlay_${token!.kind.name}_${token.label}',
                                    ),
                                    padding: const EdgeInsets.only(bottom: 12),
                                    child: Semantics(
                                      liveRegion: true,
                                      child: DecoratedBox(
                                        decoration: BoxDecoration(
                                          color: const Color(0xE01C1C1E),
                                          borderRadius:
                                              BorderRadius.circular(18),
                                          border: Border.all(
                                            color: VisualinColors.primaryOrange,
                                            width: 1.5,
                                          ),
                                        ),
                                        child: Padding(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 20,
                                            vertical: 12,
                                          ),
                                          child: Row(
                                            mainAxisSize: MainAxisSize.min,
                                            children: [
                                              const Icon(
                                                Icons.check_circle_rounded,
                                                color: VisualinColors
                                                    .primaryOrange,
                                                size: 20,
                                              ),
                                              const SizedBox(width: 10),
                                              Flexible(
                                                child: Text(
                                                  label,
                                                  textAlign: TextAlign.center,
                                                  style: Theme.of(context)
                                                      .textTheme
                                                      .titleLarge
                                                      ?.copyWith(
                                                        color: Colors.white,
                                                        fontWeight:
                                                            FontWeight.w700,
                                                      ),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      ),
                                    ),
                                  ),
                          );
                        },
                      ),
                      _FullScreenTranscriptBar(
                        overlay: _overlay,
                        hudListenable: _hudNotifier,
                        l10n: l10n,
                        session: () => _session,
                        onClear: _clearTranslation,
                        onCopy: (text) => _copyTranslation(l10n, text),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FullScreenTranscriptBar extends StatelessWidget {
  const _FullScreenTranscriptBar({
    required this.overlay,
    required this.hudListenable,
    required this.l10n,
    required this.session,
    required this.onClear,
    required this.onCopy,
  });

  final LandmarkOverlay overlay;
  final Listenable hudListenable;
  final AppLocalizations l10n;
  final ContinuousSignSession? Function() session;
  final VoidCallback onClear;
  final ValueChanged<String> onCopy;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: hudListenable,
      builder: (context, _) {
        final currentSession = session();
        final accumulated = currentSession?.formatText(l10n) ?? '';
        final hasTokens = currentSession?.hasTokens ?? false;
        final hasHand = overlay.hands.isNotEmpty;
        final status = hasHand ? l10n.handDetected : l10n.noHandDetected;
        final output = accumulated.isNotEmpty
            ? accumulated
            : (currentSession?.retry == true
                ? l10n.signRetry
                : l10n.translationWaiting);
        final badgePrediction = currentSession?.activeBadgePrediction;

        return Semantics(
          liveRegion: true,
          label: '${l10n.translatedText}. $status. $output',
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 10, 12, 12),
            decoration: BoxDecoration(
              color: const Color(0xCC1C1C1E),
              borderRadius: VisualinRadii.card,
              border: Border.all(
                color: Colors.white.withValues(alpha: 0.18),
              ),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      hasHand
                          ? Icons.check_circle_outline
                          : Icons.pan_tool_outlined,
                      size: 16,
                      color: hasHand
                          ? VisualinColors.primaryOrange
                          : Colors.white70,
                    ),
                    const SizedBox(width: 6),
                    Flexible(
                      child: Text(
                        status,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: Theme.of(context).textTheme.labelSmall?.copyWith(
                              color: hasHand
                                  ? VisualinColors.primaryOrange
                                  : Colors.white70,
                              fontWeight: FontWeight.w600,
                            ),
                      ),
                    ),
                    if (badgePrediction != null) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: VisualinColors.primaryOrange
                              .withValues(alpha: 0.22),
                          borderRadius:
                              const BorderRadius.all(Radius.circular(10)),
                        ),
                        child: Text(
                          '${l10n.signLabel(badgePrediction.label)} ${(badgePrediction.score * 100).round()}%',
                          style:
                              Theme.of(context).textTheme.labelSmall?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700,
                                  ),
                        ),
                      ),
                    ],
                    const Spacer(),
                    if (hasTokens) ...[
                      IconButton(
                        constraints: const BoxConstraints(
                          minWidth: 44,
                          minHeight: 44,
                        ),
                        padding: EdgeInsets.zero,
                        color: Colors.white,
                        tooltip: l10n.copy,
                        onPressed: () => onCopy(accumulated),
                        icon: const Icon(Icons.copy_outlined, size: 20),
                      ),
                      IconButton(
                        constraints: const BoxConstraints(
                          minWidth: 44,
                          minHeight: 44,
                        ),
                        padding: EdgeInsets.zero,
                        color: Colors.white,
                        tooltip: l10n.clearText,
                        onPressed: onClear,
                        icon: const Icon(Icons.delete_outline, size: 20),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 6),
                ConstrainedBox(
                  constraints: const BoxConstraints(maxHeight: 82),
                  child: SingleChildScrollView(
                    reverse: true,
                    child: Text(
                      output,
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            color: accumulated.isNotEmpty
                                ? Colors.white
                                : Colors.white70,
                          ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _CameraPreviewPanel extends StatelessWidget {
  const _CameraPreviewPanel({
    required this.controller,
    required this.overlay,
    required this.initializing,
    required this.cameraUnavailable,
    required this.frameLabel,
    this.overlayStyle = TrackingOverlayStyle.contour,
    this.highContrast = false,
    this.isFullScreen = false,
    this.fullScreenLabel,
    this.onEnterFullScreen,
  });

  final CameraController? controller;
  final LandmarkOverlay overlay;
  final bool initializing;
  final String cameraUnavailable;
  final String frameLabel;
  final TrackingOverlayStyle overlayStyle;
  final bool highContrast;
  final bool isFullScreen;
  final String? fullScreenLabel;
  final VoidCallback? onEnterFullScreen;

  double _safeRawAspectRatio(CameraValue value) {
    final previewSize = value.previewSize;
    if (previewSize != null && previewSize.height > 0 && previewSize.width > 0) {
      return value.aspectRatio;
    }
    return 4 / 3;
  }

  DeviceOrientation _applicableOrientation(CameraValue value) {
    if (value.isRecordingVideo && value.recordingOrientation != null) {
      return value.recordingOrientation!;
    }
    return value.previewPauseOrientation ??
        value.lockedCaptureOrientation ??
        value.deviceOrientation;
  }

  Widget _buildCoverCameraPreview(CameraController active) {
    return ValueListenableBuilder<CameraValue>(
      valueListenable: active,
      builder: (context, value, _) {
        final preview = CameraPreview(
          active,
          child: IgnorePointer(
            child: RepaintBoundary(
              child: CustomPaint(
                painter: LandmarkPainter(
                  overlay: overlay,
                  controller: active,
                  style: overlayStyle,
                  highContrast: highContrast,
                ),
              ),
            ),
          ),
        );
        return LayoutBuilder(
          builder: (context, constraints) {
            final viewportSize = Size(
              constraints.maxWidth,
              constraints.maxHeight,
            );
            final coverSize = value.isInitialized
                ? computeCameraCoverSize(
                    viewportSize: viewportSize,
                    rawAspectRatio: _safeRawAspectRatio(value),
                    orientation: _applicableOrientation(value),
                  )
                : Size.zero;
            if (coverSize == Size.zero) {
              return Center(child: preview);
            }
            return ClipRect(
              child: SizedBox.expand(
                child: OverflowBox(
                  alignment: Alignment.center,
                  minWidth: coverSize.width,
                  maxWidth: coverSize.width,
                  minHeight: coverSize.height,
                  maxHeight: coverSize.height,
                  child: SizedBox(
                    width: coverSize.width,
                    height: coverSize.height,
                    child: preview,
                  ),
                ),
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final active = controller;
    final isReady = active?.value.isInitialized == true;
    return Semantics(
      label: frameLabel,
      child: Container(
        width: double.infinity,
        decoration: isFullScreen
            ? const BoxDecoration(color: Colors.black)
            : BoxDecoration(
                color: Colors.black,
                borderRadius: VisualinRadii.card,
                border: Border.all(color: Theme.of(context).dividerColor),
              ),
        clipBehavior: Clip.antiAlias,
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (isReady)
              _buildCoverCameraPreview(active!)
            else
              Center(
                child: initializing
                    ? const CircularProgressIndicator(
                        color: VisualinColors.primaryOrange,
                      )
                    : Text(
                        cameraUnavailable,
                        style: const TextStyle(color: Colors.white),
                      ),
              ),
            if (!isFullScreen && onEnterFullScreen != null)
              Positioned(
                top: 10,
                right: 10,
                child: Material(
                  color: const Color(0xA61C1C1E),
                  shape: CircleBorder(
                    side: BorderSide(
                      color: Colors.white.withValues(alpha: 0.24),
                    ),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: IconButton(
                    constraints: const BoxConstraints(
                      minWidth: 48,
                      minHeight: 48,
                    ),
                    color: Colors.white,
                    tooltip: fullScreenLabel,
                    onPressed: onEnterFullScreen,
                    icon: const Icon(Icons.fullscreen),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _TranslationResultPanel extends StatelessWidget {
  const _TranslationResultPanel({
    required this.overlay,
    required this.hudListenable,
    required this.l10n,
    required this.session,
    required this.bodyHint,
    required this.onClear,
    required this.onCopy,
  });

  final LandmarkOverlay overlay;
  final Listenable hudListenable;
  final AppLocalizations l10n;
  final ContinuousSignSession? Function() session;
  final String? Function() bodyHint;
  final VoidCallback onClear;
  final ValueChanged<String> onCopy;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: hudListenable,
      builder: (context, _) {
        final currentSession = session();
        final accumulated = currentSession?.formatText(l10n) ?? '';
        final badgePrediction = currentSession?.activeBadgePrediction;
        final output = accumulated.isNotEmpty
            ? accumulated
            : (currentSession?.retry == true
                ? l10n.signRetry
                : l10n.translationWaiting);
        final hasHand = overlay.hands.isNotEmpty;
        final status = hasHand ? l10n.handDetected : l10n.noHandDetected;
        final guidance = bodyHint() ??
            (currentSession?.retry == true
                ? l10n.continuousRetry
                : (hasHand
                    ? l10n.cameraGuidanceHandDetected
                    : l10n.cameraGuidanceNoHand));
        final outputLanguage = l10n.isEnglish ? l10n.english : l10n.filipino;
        final hasTokens = currentSession?.hasTokens ?? false;
        return Semantics(
          liveRegion: true,
          label: '$status. $output',
          child: SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(
                  spacing: 10,
                  runSpacing: 4,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    Text(l10n.translatedText,
                        style: Theme.of(context).textTheme.titleMedium),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: const BoxDecoration(
                        color: VisualinColors.orangeTint,
                        borderRadius: BorderRadius.all(Radius.circular(12)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            hasHand
                                ? Icons.check_circle_outline
                                : Icons.pan_tool_outlined,
                            size: 16,
                            color: VisualinColors.primaryOrange,
                          ),
                          const SizedBox(width: 4),
                          Flexible(
                              child: Text(
                            status,
                            style: Theme.of(context)
                                .textTheme
                                .labelSmall
                                ?.copyWith(
                                  color: VisualinColors.primaryOrange,
                                  fontWeight: FontWeight.w600,
                                ),
                          )),
                        ],
                      ),
                    ),
                    if (badgePrediction != null)
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: const BoxDecoration(
                          color: VisualinColors.orangeTint,
                          borderRadius: BorderRadius.all(Radius.circular(12)),
                        ),
                        child: Text(
                          '${l10n.signLabel(badgePrediction.label)} ${(badgePrediction.score * 100).round()}%',
                          style:
                              Theme.of(context).textTheme.labelSmall?.copyWith(
                                    color: VisualinColors.primaryOrange,
                                    fontWeight: FontWeight.w700,
                                  ),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(14),
                  decoration: const BoxDecoration(
                    color: VisualinColors.orangeTint,
                    borderRadius: VisualinRadii.control,
                  ),
                  constraints: const BoxConstraints(maxHeight: 110),
                  child: SingleChildScrollView(
                      reverse: true,
                      child: Text(
                        output,
                        style: Theme.of(context).textTheme.bodyLarge,
                      )),
                ),
                const SizedBox(height: 8),
                Text(guidance, style: Theme.of(context).textTheme.bodyMedium),
                const SizedBox(height: 12),
                Text(
                  l10n.fslToOutputLanguage(outputLanguage),
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    OutlinedButton.icon(
                      onPressed: hasTokens ? onClear : null,
                      icon: const Icon(Icons.delete_outline),
                      label: Text(l10n.clearText),
                    ),
                    OutlinedButton.icon(
                      onPressed: hasTokens ? () => onCopy(accumulated) : null,
                      icon: const Icon(Icons.copy_outlined),
                      label: Text(l10n.copy),
                    ),
                    IconButton(
                      constraints:
                          const BoxConstraints(minWidth: 44, minHeight: 44),
                      tooltip: l10n.datasetCredit,
                      icon: const Icon(Icons.info_outline),
                      onPressed: () => showAboutDialog(
                        context: context,
                        applicationName: 'Visualin',
                        children: [
                          SelectableText(l10n.datasetCreditBody),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class LandmarkPainter extends CustomPainter {
  LandmarkPainter({
    required this.overlay,
    required this.controller,
    this.style = TrackingOverlayStyle.contour,
    this.highContrast = false,
  }) : super(repaint: Listenable.merge(<Listenable>[overlay, controller]));

  final LandmarkOverlay overlay;
  final CameraController controller;
  final TrackingOverlayStyle style;
  final bool highContrast;

  @override
  void paint(Canvas canvas, Size size) {
    final transform = CameraOverlayTransform.fromCamera(controller);
    final handsSource = overlay.displayHands;
    final poseSource = overlay.displayPose;

    final transformedHands = [
      for (final hand in handsSource)
        hand
            .map((landmark) => transform.apply(landmark.x, landmark.y))
            .map((point) => Offset(point.dx * size.width, point.dy * size.height))
            .toList(growable: false),
    ];

    final transformedPose = poseSource.length >= 5
        ? [
            for (final p in poseSource)
              Offset(
                (transform.mirrorHorizontally ? 1 - p.x : p.x) * size.width,
                p.y * size.height,
              ),
          ]
        : const <Offset>[];

    // Snap pose wrists onto matching tracked hand wrists for exact joint union
    // in both contour and classic skeleton views.
    if (transformedPose.length >= 7 && transformedHands.isNotEmpty) {
      final maxSnapDist = 0.25 * size.shortestSide;
      for (final wristIndex in const <int>[5, 6]) {
        final poseWrist = transformedPose[wristIndex];
        Offset? nearestWrist;
        var bestDist = maxSnapDist;
        for (final handPoints in transformedHands) {
          if (handPoints.isEmpty) continue;
          final dist = (handPoints.first - poseWrist).distance;
          if (dist <= bestDist) {
            bestDist = dist;
            nearestWrist = handPoints.first;
          }
        }
        if (nearestWrist != null) {
          transformedPose[wristIndex] = nearestWrist;
        }
      }
    }

    LandmarkContourRenderer.paint(
      canvas: canvas,
      size: size,
      hands: transformedHands,
      posePoints: transformedPose,
      style: style,
      highContrast: highContrast,
    );
  }

  @override
  bool shouldRepaint(LandmarkPainter oldDelegate) =>
      overlay != oldDelegate.overlay ||
      controller != oldDelegate.controller ||
      style != oldDelegate.style ||
      highContrast != oldDelegate.highContrast;
}

class _HudStateNotifier extends ChangeNotifier {
  bool _hasHands = false;
  bool _hasPose = false;
  bool _poseError = false;
  int _tokenCount = 0;
  RecognizedTokenKind? _lastTokenKind;
  String _lastTokenLabel = '';
  String? _badgeLabel;
  int? _badgePercent;
  bool _retry = false;

  void updateIfChanged({
    required bool hasHands,
    required bool hasPose,
    required bool poseError,
    required int tokenCount,
    required RecognizedTokenKind? lastTokenKind,
    required String lastTokenLabel,
    required String? badgeLabel,
    required int? badgePercent,
    required bool retry,
  }) {
    if (hasHands == _hasHands &&
        hasPose == _hasPose &&
        poseError == _poseError &&
        tokenCount == _tokenCount &&
        lastTokenKind == _lastTokenKind &&
        lastTokenLabel == _lastTokenLabel &&
        badgeLabel == _badgeLabel &&
        badgePercent == _badgePercent &&
        retry == _retry) {
      return;
    }
    _hasHands = hasHands;
    _hasPose = hasPose;
    _poseError = poseError;
    _tokenCount = tokenCount;
    _lastTokenKind = lastTokenKind;
    _lastTokenLabel = lastTokenLabel;
    _badgeLabel = badgeLabel;
    _badgePercent = badgePercent;
    _retry = retry;
    notifyListeners();
  }
}
