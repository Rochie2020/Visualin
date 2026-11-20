import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:visualin/l10n/app_localizations.dart';
import 'package:visualin/screens/camera_translation_screen.dart';
import 'package:visualin/screens/profile_screen.dart';
import 'package:visualin/services/alphabet_classifier.dart';
import 'package:visualin/services/alphabet_sign_capture.dart';
import 'package:visualin/services/app_preferences.dart';
import 'package:visualin/services/continuous_sign_session.dart';
import 'package:visualin/services/landmark_service.dart';
import 'package:visualin/services/sign_capture.dart';
import 'package:visualin/services/sign_classifier.dart';
import 'package:visualin/widgets/app_scaffold.dart';
import 'package:visualin/widgets/landmark_contour_painter.dart';
import 'package:visualin/widgets/landmark_overlay.dart';

class _FakeLandmarkDetector implements LandmarkDetector {
  final StreamController<LandmarkFrame> _controller =
      StreamController<LandmarkFrame>.broadcast();
  bool bodyTrackingEnabled = false;

  void emit(LandmarkFrame frame) => _controller.add(frame);

  @override
  Stream<LandmarkFrame> get frames => _controller.stream;

  @override
  void setBodyTracking(bool enabled) {
    bodyTrackingEnabled = enabled;
  }

  @override
  Future<void> start(
    CameraController controller, {
    bool Function()? isActive,
  }) async {}

  @override
  Future<void> stop() async {}

  Future<void> close() => _controller.close();
}

class _FakeCameraController extends CameraController {
  _FakeCameraController(CameraDescription cameraDescription)
      : super(
          cameraDescription,
          ResolutionPreset.medium,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.yuv420,
        );

  @override
  Future<void> initialize() async {
    value = value.copyWith(
      isInitialized: true,
      previewSize: const Size(640, 480),
      deviceOrientation: DeviceOrientation.portraitUp,
      description: description,
    );
  }

  @override
  Widget buildPreview() => const ColoredBox(color: Color(0xFF102030));

  @override
  Future<void> stopImageStream() async {}

  @override
  // ignore: must_call_super
  Future<void> dispose() async {
    // Avoid platform channel disposal in headless widget tests.
  }
}

List<LandmarkPoint> _loadAlphabetSamplePoints(String label) {
  final raw = File('assets/ml/alphabet_preprocessing.json').readAsStringSync();
  final decoded = jsonDecode(raw) as Map<String, dynamic>;
  final samples = List<Map<String, dynamic>>.from(
    decoded['testSamples'] as List,
  );
  final sample = samples.firstWhere((s) => s['label'] == label);
  return [
    for (final p in sample['landmarks'] as List)
      LandmarkPoint(
        (p[0] as num).toDouble(),
        (p[1] as num).toDouble(),
        (p[2] as num).toDouble(),
      ),
  ];
}

ContinuousSignSession _loadTestSessionSync() {
  final txModel = SignClassifier.decode(
    File('assets/ml/transaction_signs.json').readAsStringSync(),
  );
  final combined = SignClassifier.decode(
    File('assets/ml/fsl_combined_signs.json').readAsStringSync(),
  ).withFallback(txModel);
  final alphaConfig = jsonDecode(
    File('assets/ml/alphabet_preprocessing.json').readAsStringSync(),
  ) as Map<String, dynamic>;
  final alphaModel = AlphabetClassifier.fromTfliteBytes(
    File('assets/ml/alphabet_classifier.tflite').readAsBytesSync(),
    labels: List<String>.from(alphaConfig['labels'] as List),
    threshold: (alphaConfig['threshold'] as num).toDouble(),
  );
  return ContinuousSignSession(
    dynamicCapture: SignCapture(combined),
    alphabetCapture: AlphabetSignCapture(alphaModel),
  );
}

void main() {
  testWidgets(
      'CameraTranslationScreen hides Undo/Space and obsolete toggles while rendering unified controls',
      (tester) async {
    final detector = _FakeLandmarkDetector();
    addTearDown(detector.close);

    await tester.pumpWidget(
      MaterialApp(
        localizationsDelegates: const [
          ...AppLocalizations.localizationsDelegates,
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        supportedLocales: AppLocalizations.supportedLocales,
        home: CameraTranslationScreen(
          landmarkService: detector,
          sessionLoader: () async => _loadTestSessionSync(),
        ),
      ),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(find.text('FSL to Text'), findsWidgets);
    expect(find.byType(SwitchListTile), findsNothing);
    expect(find.text('Continuous signing (trial)'), findsNothing);
    expect(find.text('Upper-body tracking (42 / 104 signs)'), findsNothing);
    expect(find.text('Alphabet Mode'), findsNothing);

    expect(find.text('Undo last sign'), findsNothing);
    expect(find.text('Space'), findsNothing);
    expect(find.text('Clear'), findsOneWidget);
    expect(find.text('Copy'), findsOneWidget);
    expect(find.byTooltip('Full screen'), findsOneWidget);
    expect(find.byTooltip('FSL-105 dataset credits'), findsOneWidget);
  });

  test('cover sizing fills portrait and landscape viewports without distortion', () {
    const portraitViewport = Size(360, 720);
    final portraitCover = computeCameraCoverSize(
      viewportSize: portraitViewport,
      rawAspectRatio: 4 / 3,
      orientation: DeviceOrientation.portraitUp,
    );
    expect(portraitCover.width, greaterThanOrEqualTo(portraitViewport.width - 1e-6));
    expect(portraitCover.height, greaterThanOrEqualTo(portraitViewport.height - 1e-6));
    expect(portraitCover.width / portraitCover.height, closeTo(3 / 4, 1e-6));

    const landscapeViewport = Size(840, 360);
    final landscapeCover = computeCameraCoverSize(
      viewportSize: landscapeViewport,
      rawAspectRatio: 4 / 3,
      orientation: DeviceOrientation.landscapeLeft,
    );
    expect(landscapeCover.width, greaterThanOrEqualTo(landscapeViewport.width - 1e-6));
    expect(landscapeCover.height, greaterThanOrEqualTo(landscapeViewport.height - 1e-6));
    expect(landscapeCover.width / landscapeCover.height, closeTo(4 / 3, 1e-6));
  });

  testWidgets(
      'cover preview aligns landmarks cleanly and full-screen briefly overlays confirmed signs while keeping accumulated text',
      (tester) async {
    final detector = _FakeLandmarkDetector();
    addTearDown(detector.close);
    final handB = _loadAlphabetSamplePoints('B');
    final handO = _loadAlphabetSamplePoints('O');
    final session = _loadTestSessionSync();
    var nowMs = 0;

    const fakeCamera = CameraDescription(
      name: '0',
      lensDirection: CameraLensDirection.front,
      sensorOrientation: 270,
    );

    await tester.pumpWidget(
      MaterialApp(
        localizationsDelegates: const [
          ...AppLocalizations.localizationsDelegates,
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        supportedLocales: AppLocalizations.supportedLocales,
        home: CameraTranslationScreen(
          landmarkService: detector,
          camerasLoader: () async => const [fakeCamera],
          controllerFactory: _FakeCameraController.new,
          sessionLoader: () async => session,
          overlayDuration: const Duration(milliseconds: 600),
          frameNow: () => nowMs,
        ),
      ),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    // Verify full-bleed cover preview + LandmarkPainter fills inner viewport cleanly without black letterbox gaps.
    final cameraPreviewSize = tester.getSize(find.byType(CameraPreview));
    final previewViewportFinder = find.descendant(
      of: find.bySemanticsLabel('Hand position guide'),
      matching: find.byType(ClipRect),
    );
    final previewContainerSize = tester.getSize(previewViewportFinder);
    expect(cameraPreviewSize.width, greaterThanOrEqualTo(previewContainerSize.width - 0.5));
    expect(cameraPreviewSize.height, greaterThanOrEqualTo(previewContainerSize.height - 0.5));
    expect(
      cameraPreviewSize.width / cameraPreviewSize.height,
      closeTo(3 / 4, 1e-3),
    );

    // Confirm 'B' in normal view.
    for (var i = 0; i < 6; i++) {
      nowMs += 100;
      detector.emit(LandmarkFrame([handB], rotation: 0, aspect: 1.0));
      await tester.pump();
    }
    expect(find.text('B'), findsWidgets);

    // Enter full-screen view.
    await tester.tap(find.byTooltip('Full screen'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 250));

    expect(find.byType(VisualinBottomNav), findsNothing);
    expect(find.text('Exit full screen'), findsOneWidget);
    expect(find.text('Undo last sign'), findsNothing);
    expect(find.text('Space'), findsNothing);
    expect(find.text('B'), findsWidgets);

    // Full-screen cover camera fills entire screen without stretching.
    final fullCameraSize = tester.getSize(find.byType(CameraPreview));
    final fullScreenSize = tester.getSize(
      find.descendant(
        of: find.bySemanticsLabel('Hand position guide'),
        matching: find.byType(ClipRect),
      ),
    );
    expect(fullCameraSize.width, greaterThanOrEqualTo(fullScreenSize.width - 0.5));
    expect(fullCameraSize.height, greaterThanOrEqualTo(fullScreenSize.height - 0.5));
    expect(fullCameraSize.width / fullCameraSize.height, closeTo(3 / 4, 1e-3));

    // Confirm 'O' while in full-screen -> transient overlay shows 'O' cleanly.
    for (var i = 0; i < 5; i++) {
      nowMs += 100;
      detector.emit(LandmarkFrame([handO], rotation: 0, aspect: 1.0));
      await tester.pump();
    }
    expect(find.byKey(const ValueKey('overlay_letter_O')), findsOneWidget);
    expect(find.text('BO'), findsOneWidget);

    // Wait past overlayDuration -> temporary overlay disappears, accumulated 'BO' stays preserved!
    await tester.pump(const Duration(milliseconds: 650));
    await tester.pump(const Duration(milliseconds: 250));
    expect(find.byKey(const ValueKey('overlay_letter_O')), findsNothing);
    expect(find.text('BO'), findsOneWidget);

    // Exit full-screen back to normal view -> accumulated text remains intact!
    await tester.tap(find.text('Exit full screen'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 250));
    expect(find.byType(VisualinBottomNav), findsOneWidget);
    expect(find.text('BO'), findsOneWidget);
  });

  test('new continuous editing and full-screen labels localize in English and Filipino', () {
    const en = AppLocalizations(Locale('en'));
    const fil = AppLocalizations(Locale('fil'));

    expect(en.clearText, 'Clear');
    expect(en.space, 'Space');
    expect(en.fullScreen, 'Full screen');
    expect(en.exitFullScreen, 'Exit full screen');
    expect(en.copiedToClipboard, 'Copied translated text');
    expect(fil.clearText, 'Burahin');
    expect(fil.space, 'Espasyo');
    expect(fil.fullScreen, 'Buong screen');
    expect(fil.exitFullScreen, 'Lumabas sa buong screen');
    expect(fil.copiedToClipboard, 'Nakopya ang isinaling teksto');
    expect(en.skeletonOverlayMode, 'Classic skeleton overlay');
    expect(fil.skeletonOverlayMode, 'Klasikong skeleton overlay');
    expect(en.switchToSkeletonOverlay, 'Switch to skeleton overlay');
    expect(fil.switchToContourOverlay, 'Lumipat sa contour overlay');
  });

  test('contour renderer builds smooth hand outlines, head corner rect, and paints both styles cleanly', () {
    const size = Size(400, 600);
    // 11-point pose: nose, L/R shoulder, L/R elbow, L/R wrist, L/R ear, L/R hip.
    final pose11 = <Offset>[
      const Offset(200, 120), // nose
      const Offset(145, 200), // L shoulder
      const Offset(255, 200), // R shoulder
      const Offset(120, 290), // L elbow
      const Offset(280, 290), // R elbow
      const Offset(115, 360), // L wrist
      const Offset(285, 360), // R wrist
      const Offset(172, 114), // L ear
      const Offset(228, 114), // R ear
      const Offset(160, 345), // L hip
      const Offset(240, 345), // R hip
    ];
    final headRect =
        LandmarkContourRenderer.computeHeadTrackingRect(pose11, size);
    expect(headRect, isNotNull);
    expect(headRect!.contains(pose11[0]), isTrue);
    expect(headRect.width, greaterThanOrEqualTo(55));
    expect(headRect.height / headRect.width, closeTo(1.24, 0.05));

    // Also verify graceful 7-point pose fallback when ears/hips are absent.
    final fallbackHead = LandmarkContourRenderer.computeHeadTrackingRect(
      pose11.take(7).toList(),
      size,
    );
    expect(fallbackHead, isNotNull);
    expect(fallbackHead!.contains(pose11[0]), isTrue);

    List<Offset> toOffsets(List<LandmarkPoint> h) => [
          for (final p in h) Offset(p.x * size.width, p.y * size.height),
        ];

    final openHand = toOffsets(_loadAlphabetSamplePoints('B'));
    final fistHand = toOffsets(_loadAlphabetSamplePoints('A'));
    final mirroredOpenHand = [
      for (final p in openHand) Offset(size.width - p.dx, p.dy),
    ];

    for (final pts in [openHand, fistHand, mirroredOpenHand]) {
      final built = LandmarkContourRenderer.buildHandContourPaths(pts);
      expect(built.outline.getBounds().isEmpty, isFalse);
      expect(built.collar.getBounds().width, greaterThan(2.0));
    }

    for (final style in TrackingOverlayStyle.values) {
      for (final hc in const [false, true]) {
        final recorder = ui.PictureRecorder();
        final canvas = ui.Canvas(recorder);
        LandmarkContourRenderer.paint(
          canvas: canvas,
          size: size,
          hands: [openHand, fistHand],
          posePoints: pose11,
          style: style,
          highContrast: hc,
        );
        final picture = recorder.endRecording();
        expect(picture, isNotNull);
        picture.dispose();
      }
    }
  });

  test('LandmarkOverlay stabilizes jitter cleanly while decoupling statusRevision from frame noise', () {
    final overlay = LandmarkOverlay();
    addTearDown(overlay.dispose);
    final handB = _loadAlphabetSamplePoints('B');

    var statusEvents = 0;
    overlay.statusRevision.addListener(() => statusEvents++);

    overlay.update(LandmarkFrame([handB], rotation: 0, aspect: 1.0));
    expect(statusEvents, 1); // empty -> present transitions once
    expect(overlay.displayHands, equals([handB]));

    // Tiny micro-jitter (~0.001) should be damped rather than snapping 1:1.
    final jittered = [
      for (final p in handB) LandmarkPoint(p.x + 0.001, p.y - 0.001, p.z),
    ];
    overlay.update(LandmarkFrame([jittered], rotation: 0, aspect: 1.0));
    expect(statusEvents, 1); // no redundant status rebuild when hands remain visible!
    expect(overlay.hands.first.first.x, closeTo(jittered.first.x, 1e-6));
    expect(
      overlay.displayHands.first.first.x,
      lessThan(jittered.first.x),
    );

    // Fast intentional motion (>= 0.10 palm shift) responds near-immediately (>90% gain).
    final fastMoved = [
      for (final p in handB) LandmarkPoint(p.x + 0.03, p.y, p.z),
    ];
    final prevDisplayX = overlay.displayHands.first.first.x;
    overlay.update(LandmarkFrame([fastMoved], rotation: 0, aspect: 1.0));
    final gainRatio =
        (overlay.displayHands.first.first.x - prevDisplayX) /
            (fastMoved.first.x - prevDisplayX);
    expect(gainRatio, greaterThan(0.90));
  });

  testWidgets('overlay style toggles between modern contour and skeleton and persists to preferences', (tester) async {
    SharedPreferences.setMockInitialValues(<String, Object>{});
    final preferences = await AppPreferences.create();
    expect(preferences.overlayStyle, TrackingOverlayStyle.contour);

    final detector = _FakeLandmarkDetector();
    addTearDown(detector.close);

    await tester.pumpWidget(
      MaterialApp(
        localizationsDelegates: const [
          ...AppLocalizations.localizationsDelegates,
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        supportedLocales: AppLocalizations.supportedLocales,
        home: CameraTranslationScreen(
          landmarkService: detector,
          preferences: preferences,
          camerasLoader: () async => const [],
          sessionLoader: () async => _loadTestSessionSync(),
        ),
      ),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(find.byType(SwitchListTile), findsNothing);
    expect(find.byTooltip('Switch to skeleton overlay'), findsOneWidget);

    await tester.tap(find.byTooltip('Switch to skeleton overlay'));
    await tester.pump();

    expect(preferences.overlayStyle, TrackingOverlayStyle.skeleton);
    expect(find.byTooltip('Switch to contour overlay'), findsOneWidget);

    // Toggle also works inside full-screen view and stays persisted.
    await tester.tap(find.byTooltip('Full screen'));
    await tester.pump();
    expect(find.byTooltip('Switch to contour overlay'), findsOneWidget);
    await tester.tap(find.byTooltip('Switch to contour overlay'));
    await tester.pump();
    expect(preferences.overlayStyle, TrackingOverlayStyle.contour);
    await tester.tap(find.byTooltip('Switch to skeleton overlay'));
    await tester.pump();
    expect(preferences.overlayStyle, TrackingOverlayStyle.skeleton);
    await tester.tap(find.text('Exit full screen'));
    await tester.pump();

    final reloaded = await AppPreferences.create();
    expect(reloaded.overlayStyle, TrackingOverlayStyle.skeleton);

    // ProfileScreen exposes the same setting in English and switches cleanly back to contour.
    await tester.pumpWidget(
      MaterialApp(
        localizationsDelegates: const [
          ...AppLocalizations.localizationsDelegates,
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        supportedLocales: AppLocalizations.supportedLocales,
        home: ProfileScreen(
          preferences: reloaded,
          onLocaleChanged: (_) async {},
          onHighContrastChanged: (_) async {},
          onReducedMotionChanged: (_) async {},
          textScale: 1.0,
          onTextScaleChanged: (_) async {},
        ),
      ),
    );
    await tester.pumpAndSettle();

    final skeletonTile = find.text('Classic skeleton overlay');
    await tester.scrollUntilVisible(skeletonTile, 120);
    await tester.ensureVisible(skeletonTile);
    await tester.pumpAndSettle();
    expect(skeletonTile, findsOneWidget);
    await tester.tap(skeletonTile);
    await tester.pumpAndSettle();
    expect(reloaded.overlayStyle, TrackingOverlayStyle.contour);
  });
}
