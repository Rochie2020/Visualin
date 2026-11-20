import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' show Locale;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:visualin/l10n/app_localizations.dart';
import 'package:visualin/services/alphabet_classifier.dart';
import 'package:visualin/services/alphabet_sign_capture.dart';
import 'package:visualin/services/continuous_sign_capture.dart';
import 'package:visualin/services/continuous_sign_session.dart';
import 'package:visualin/services/landmark_service.dart';
import 'package:visualin/services/sign_capture.dart';
import 'package:visualin/services/sign_classifier.dart';
import 'package:visualin/widgets/camera_overlay_transform.dart';

List<SignFrame> _readClip(File file) {
  final bytes = file.readAsBytesSync();
  final data = ByteData.sublistView(bytes);
  final start = 10 + data.getUint16(8, Endian.little);
  expect(bytes.length, start + 75 * 225 * 8);
  final frames = <SignFrame>[];
  for (var frame = 0; frame < 75; frame += 2) {
    final hands = <List<SignPoint>>[];
    for (var hand = 0; hand < 2; hand++) {
      final points = <SignPoint>[];
      for (var j = 0; j < 21; j++) {
        final offset = start + (frame * 225 + 99 + hand * 63 + j * 3) * 8;
        points.add((
          x: data.getFloat64(offset, Endian.little) * 4 / 3,
          y: data.getFloat64(offset + 8, Endian.little)
        ));
      }
      if (points.any((p) => p.x != 0 || p.y != 0)) hands.add(points);
    }
    frames.add(hands);
  }
  return frames;
}

// Put a known upright sign in the raw space of a portrait camera.
// Translation and the uniform zoom represent different framing on the phone.
LandmarkFrame _sensorFrame(SignFrame frame, int rotation, bool left) {
  const aspect = 3 / 4;
  final raw = <List<LandmarkPoint>>[];
  for (final hand in frame.reversed) {
    raw.add(hand.map((p) {
      final x = (left ? 1 - p.x * .55 : p.x * .55) + .03;
      final y = p.y * .55 * aspect + .12;
      return rotation == 90
          ? LandmarkPoint(y, 1 - x, 0)
          : LandmarkPoint(1 - y, x, 0);
    }).toList());
  }
  return LandmarkFrame(raw, rotation: rotation, aspect: aspect);
}

void main() {
  final classifier = SignClassifier.decode(
      File('assets/ml/transaction_signs.json').readAsStringSync());
  test('each trained label has English and Filipino output', () {
    for (final label in classifier.labels) {
      for (final language in ['en', 'fil']) {
        final text = AppLocalizations(Locale(language)).signLabel(label);
        expect(text, isNotEmpty);
      }
    }
    expect(const AppLocalizations(Locale('fil')).signLabel('HOW_MUCH'),
        'Magkano?');
  });

  test(
      'forearm display endpoints preserve model geometry and frozen Hello is rejected',
      () {
    final payload =
        jsonDecode(File('assets/ml/body_signs.json').readAsStringSync())
            as Map<String, dynamic>;
    final model = SignClassifier.decode(jsonEncode(payload));
    payload.remove('motionFloors');
    final before = SignClassifier.decode(jsonEncode(payload));
    final files =
        Directory('coordinates/HELLO').listSync().whereType<File>().toList();
    final bytes =
        File('coordinates/HELLO/HELLO_${files.length}.npy').readAsBytesSync();
    final data = ByteData.sublistView(bytes);
    final start = 10 + data.getUint16(8, Endian.little);
    final frames = <BodySignFrame>[];
    for (var frame = 0; frame < 75; frame += 2) {
      LandmarkPoint read(int index) {
        final offset = start + (frame * 225 + index * 3) * 8;
        return LandmarkPoint(
            data.getFloat64(offset, Endian.little),
            data.getFloat64(offset + 8, Endian.little),
            data.getFloat64(offset + 16, Endian.little));
      }

      final pose = [
        for (final i in [0, 11, 12, 13, 14, 15, 16]) read(i)
      ];
      final hands = [
        for (final base in [33, 54])
          [for (var i = 0; i < 21; i++) read(base + i)]
      ].where((hand) => hand.any((p) => p.x != 0 || p.y != 0)).toList();
      if (hands.isEmpty) continue;
      final input = LandmarkFrame(hands, pose: pose, aspect: 4 / 3).modelBody;
      expect(input.pose.length, 5);
      frames.add(input);
    }
    expect(model.predict(frames)?.label, 'HELLO');
    final index = frames.indexWhere(
        (f) => before.predict(List.filled(12, f))?.label == 'HELLO');
    expect(index, greaterThanOrEqualTo(0));
    expect(model.predict(List.filled(12, frames[index])), isNull);
    expect(model.motionFloors.containsKey('FIVE'), isFalse);
  });

  test(
      'the seven greeting labels have distinct localized output and extend the trial to 42',
      () {
    final base = SignClassifier.decode(
        File('assets/ml/body_signs.json').readAsStringSync());
    final specialist = SignClassifier.decode(
        File('assets/ml/greeting_signs.json').readAsStringSync());
    final trial = base.withGreetings(specialist);
    final all = {...trial.labels, ...specialist.labels}..remove('__OTHER__');
    expect(all.length, 42);
    expect(specialist.labels.toSet(), {
      'GOOD_MORNING',
      'GOOD_AFTERNOON',
      'HOW_ARE_YOU',
      'IM_FINE',
      'NICE_TO_MEET_YOU',
      'YOURE_WELCOME',
      'SEE_YOU_TOMORROW',
      '__OTHER__'
    });
    for (final locale in ['en', 'fil']) {
      final strings = AppLocalizations(Locale(locale));
      expect(strings.signLabel('YOURE_WELCOME'),
          locale == 'en' ? "You're welcome" : 'Walang anuman');
      expect(strings.signLabel('YOURE_WELCOME'),
          isNot(strings.signLabel('WELCOME')));
      expect(strings.bodyMode, contains('42'));
    }
  });

  test('upper-body trial localizes all 41 words and requires a body reference',
      () {
    final model = SignClassifier.decode(
        File('assets/ml/body_signs.json').readAsStringSync());
    final words = model.labels.where((label) => label != '__OTHER__').toList();
    expect(words.length, 41);
    for (final label in words) {
      for (final language in ['en', 'fil']) {
        final text = AppLocalizations(Locale(language)).signLabel(label);
        expect(text, isNotEmpty);
        expect(text, isNot('sign_$label'));
      }
    }
    expect(const AppLocalizations(Locale('fil')).signLabel('WEDNESDAY'),
        'Miyerkules');
    final hands = _readClip(File('coordinates/HELLO/HELLO_1.npy'));
    expect(model.predict(hands), isNull);
    final missingPose = hands
        .map((f) => BodySignFrame(
            f, const [], [for (final hand in f) List.filled(hand.length, 0.0)]))
        .toList();
    expect(model.predict(missingPose), isNull);
  });

  test('held-out recordings survive both lens rotations and a hand change', () {
    for (final label in ['HELLO', 'TWO', 'THANK_YOU', 'COIN']) {
      final files =
          Directory('coordinates/$label').listSync().whereType<File>().toList();
      final clip =
          _readClip(File('coordinates/$label/${label}_${files.length}.npy'));
      final baseline = classifier.predict(clip)!;
      expect(baseline.label, label);
      for (final front in [true, false]) {
        final rotation = CameraOverlayTransform.fromConfiguration(
          sensorOrientation: front ? 270 : 90,
          deviceOrientation: DeviceOrientation.portraitUp,
          lensDirection:
              front ? CameraLensDirection.front : CameraLensDirection.back,
        ).rotationDegrees;
        for (final left in [true, false]) {
          final frames = clip
              .map((f) => _sensorFrame(f, rotation, left).modelHands)
              .toList();
          final prediction = classifier.predict(frames)!;
          expect(prediction.label, baseline.label);
          expect(prediction.score, closeTo(baseline.score, 0.00001));
          final capture = SignCapture(classifier);
          for (var i = 0; i < frames.length; i++) {
            capture.add(frames[i], i * 80);
          }
          capture.add([], 4000);
          expect(capture.result?.label, label);
        }
      }
    }
    expect(classifier.predict([[], []]), isNull);
  });

  test('continuous mode accepts a sequence with hands still visible', () {
    final trial = SignClassifier.decode(
        File('assets/ml/continuous_signs.json').readAsStringSync());
    for (final label in trial.labels) {
      for (final language in ['en', 'fil']) {
        expect(AppLocalizations(Locale(language)).signLabel(label), isNotEmpty);
      }
    }
    for (final rotation in [90, 270]) {
      for (final mirror in [false, true]) {
        final capture = ContinuousSignCapture(trial);
        var time = 0;
        for (final label in ['TWO', 'COIN', 'TWO']) {
          final folder = Directory('coordinates/$label');
          final clip = _readClip(File(
              '${folder.path}/${label}_${folder.listSync().whereType<File>().length}.npy'));
          final visible = clip.where((f) => f.isNotEmpty);
          for (final frame in [...visible, ...List.filled(24, visible.last)]) {
            capture.add(_sensorFrame(frame, rotation, mirror).modelHands, time);
            time += 80;
          }
        }
        expect(capture.labels, ['TWO', 'COIN', 'TWO']);
        capture.resetSegment();
        expect(capture.labels.length, 3);
      }
    }
  });

  test('a short detection cannot produce text', () {
    final capture = SignCapture(classifier);
    final hand = List.generate(21, (i) => (x: i * .01, y: i * .02));
    capture.add([hand], 0);
    capture.add([], 800);
    expect(capture.result, isNull);
    expect(capture.retry, isTrue);
  });

  group('static FSL alphabet model and capture', () {
    final config = jsonDecode(
      File('assets/ml/alphabet_preprocessing.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    final labels = List<String>.from(config['labels'] as List);
    final threshold = (config['threshold'] as num).toDouble();
    final testSamples = List<Map<String, dynamic>>.from(
      config['testSamples'] as List,
    );
    final tfliteBytes =
        File('assets/ml/alphabet_classifier.tflite').readAsBytesSync();
    final alphabetClassifier = AlphabetClassifier.fromTfliteBytes(
      tfliteBytes,
      labels: labels,
      threshold: threshold,
    );

    List<LandmarkPoint> decodeSampleLandmarks(Map<String, dynamic> sample) {
      final raw = sample['landmarks'] as List;
      return [
        for (final p in raw)
          LandmarkPoint(
            (p[0] as num).toDouble(),
            (p[1] as num).toDouble(),
            (p[2] as num).toDouble(),
          ),
      ];
    }

    LandmarkFrame alphabetSensorFrame(
      List<LandmarkPoint> uprightHand,
      int rotation,
      bool mirrored,
    ) {
      const aspect = 3 / 4;
      const scale = 0.5;
      final sensorHand = uprightHand.map((p) {
        final baseX = (p.x * scale) / aspect + 0.1;
        final nx = mirrored ? 1.0 - baseX : baseX;
        final ny = p.y * scale + 0.15;
        final nz = p.z * scale / aspect;
        return switch (rotation) {
          90 => LandmarkPoint(ny, 1.0 - nx, nz),
          270 => LandmarkPoint(1.0 - ny, nx, nz),
          180 => LandmarkPoint(1.0 - nx, 1.0 - ny, nz),
          _ => LandmarkPoint(nx, ny, nz),
        };
      }).toList(growable: false);
      return LandmarkFrame([sensorHand], rotation: rotation, aspect: aspect);
    }

    List<LandmarkPoint> translateHand(
      List<LandmarkPoint> hand, {
      double dx = 0,
      double dy = 0,
    }) {
      return [
        for (final p in hand) LandmarkPoint(p.x + dx, p.y + dy, p.z),
      ];
    }

    test('alphabet classes A-Z and strings localize in English and Filipino',
        () {
      expect(labels.length, 26);
      for (final label in labels) {
        expect(const AppLocalizations(Locale('en')).signLabel(label), label);
        expect(const AppLocalizations(Locale('fil')).signLabel(label), label);
      }
      const en = AppLocalizations(Locale('en'));
      const fil = AppLocalizations(Locale('fil'));
      expect(en.alphabetMode, 'Alphabet Mode');
      expect(en.alphabetHint, isNotEmpty);
      expect(en.alphabetRetry, isNotEmpty);
      expect(en.alphabetError, isNotEmpty);
      expect(fil.alphabetMode, isNotEmpty);
      expect(fil.alphabetHint, isNotEmpty);
      expect(fil.alphabetRetry, isNotEmpty);
      expect(fil.alphabetError, isNotEmpty);
    });

    test('AlphabetClassifier.loadFromAssets loads exported model & config',
        () async {
      TestWidgetsFlutterBinding.ensureInitialized();
      final loaded = await AlphabetClassifier.loadFromAssets();
      expect(loaded.labels, labels);
      expect(loaded.threshold, threshold);
      expect(loaded.inputFeatures, AlphabetClassifier.featureCount);
    });

    test(
        'Dart TFLite output matches Python Interpreter across all 26 held-out classes',
        () {
      expect(testSamples.length, 26);
      for (final sample in testSamples) {
        final expectedLabel = sample['expectedTopLabel'] as String;
        final expectedProbs = [
          for (final v in sample['expectedProbs'] as List) (v as num).toDouble()
        ];
        final expectedFeats = [
          for (final v in sample['features'] as List) (v as num).toDouble()
        ];

        final probsFromFeats =
            alphabetClassifier.predictProbsFromFeatures(expectedFeats);
        for (var i = 0; i < labels.length; i++) {
          expect(probsFromFeats[i], closeTo(expectedProbs[i], 1e-4));
        }

        final hand = decodeSampleLandmarks(sample);
        final dartFeats = AlphabetClassifier.extractFeatures(hand)!;
        for (var i = 0; i < expectedFeats.length; i++) {
          expect(dartFeats[i], closeTo(expectedFeats[i], 1e-4));
        }

        final pred = alphabetClassifier.predictHand(hand)!;
        expect(pred.label, expectedLabel);
        expect(
          pred.score,
          closeTo((sample['expectedTopScore'] as num).toDouble(), 1e-4),
        );
      }
      final firstHand = decodeSampleLandmarks(testSamples.first);
      final sw = Stopwatch()..start();
      for (var i = 0; i < 200; i++) {
        alphabetClassifier.predictHand(firstHand);
      }
      sw.stop();
      expect(sw.elapsedMicroseconds / 200.0, lessThan(5000.0));
    });

    test(
        'left/right mirror and portrait/landscape camera rotations stay invariant',
        () {
      for (final sample in testSamples) {
        final hand = decodeSampleLandmarks(sample);
        final baseline = alphabetClassifier.predictHand(hand)!;
        for (final rotation in const <int>[0, 90, 180, 270]) {
          for (final mirrored in const <bool>[false, true]) {
            final frame = alphabetSensorFrame(hand, rotation, mirrored);
            final pred = alphabetClassifier.predict(frame.modelHands3D);
            expect(pred, isNotNull);
            expect(pred!.label, baseline.label,
                reason:
                    'Mismatch on ${sample['label']} rot=$rotation mirrored=$mirrored');
            expect(pred.score, closeTo(baseline.score, 1e-3));
          }
        }
      }
    });

    test('AlphabetSignCapture stabilizes predictions and clears stale state',
        () {
      final capture = AlphabetSignCapture(alphabetClassifier);
      final sampleA = decodeSampleLandmarks(
          testSamples.firstWhere((s) => s['label'] == 'A'));
      final sampleB = decodeSampleLandmarks(
          testSamples.firstWhere((s) => s['label'] == 'B'));

      // Single frame flicker should not confirm a letter.
      capture.add([sampleA], 0);
      expect(capture.letters, isEmpty);

      // Steady hold across >= 280 ms confirms 'A' once and latches.
      capture.add([sampleA], 100);
      capture.add([sampleA], 200);
      capture.add([sampleA], 300);
      capture.add([sampleA], 400);
      capture.add([sampleA], 500);
      expect(capture.letters, ['A']);
      expect(capture.activePrediction?.label, 'A');
      expect(capture.retry, isFalse);

      // Switching to another steady handshape confirms 'B' once stale 'A' frames expire.
      for (var t = 600; t <= 1100; t += 100) {
        capture.add([sampleB], t);
      }
      expect(capture.letters, ['A', 'B']);
      expect(capture.activePrediction?.label, 'B');

      // Missing hand beyond 450 ms clears active prediction and unlatches 'B'.
      capture.add(const [], 1200);
      expect(capture.activePrediction?.label, 'B'); // Brief gap tolerated.
      capture.add(const [], 1700);
      expect(capture.activePrediction, isNull);
      expect(capture.retry, isFalse);

      // Re-showing 'B' cleanly appends a repeated 'B'.
      for (var t = 1800; t <= 2200; t += 100) {
        capture.add([sampleB], t);
      }
      expect(capture.letters, ['A', 'B', 'B']);

      capture.undo();
      expect(capture.letters, ['A', 'B']);
      capture.clear();
      expect(capture.letters, isEmpty);
      expect(capture.activePrediction, isNull);

      // Simulate ~4.5 FPS phone CPU Landmarker intervals (220 ms/frame).
      capture.add([sampleA], 3000);
      expect(capture.activePrediction?.label, 'A');
      expect(capture.letters, isEmpty);
      capture.add([sampleA], 3220);
      expect(capture.letters, isEmpty);
      capture.add([sampleA], 3440);
      expect(capture.letters, ['A']);
    });

    test('AlphabetSignCapture rejects rapid motion, empty/bad hands, and low confidence',
        () {
      final capture = AlphabetSignCapture(alphabetClassifier);
      final sampleA = decodeSampleLandmarks(
          testSamples.firstWhere((s) => s['label'] == 'A'));
      final sampleB = decodeSampleLandmarks(
          testSamples.firstWhere((s) => s['label'] == 'B'));
      final sampleO = decodeSampleLandmarks(
          testSamples.firstWhere((s) => s['label'] == 'O'));

      // Malformed / too-short hand landmarks are safely treated as missing.
      capture.add([sampleA.take(10).toList()], 0);
      expect(capture.letters, isEmpty);

      // Large rapid translation across frames is blocked by velocity guard.
      for (var i = 0; i < 6; i++) {
        capture.add([translateHand(sampleA, dx: i * 0.25)], i * 100);
      }
      expect(capture.letters, isEmpty);

      // Steady ambiguous shape held >= 650 ms sets retry when confidence is below threshold.
      final hybridClassifier = AlphabetClassifier.fromTfliteBytes(
        tfliteBytes,
        labels: labels,
        threshold: 0.999999,
      );
      final uncertainCapture = AlphabetSignCapture(hybridClassifier);
      final blended = [
        for (var i = 0; i < 21; i++)
          LandmarkPoint(
            (sampleO[i].x + sampleB[i].x) * 0.5,
            (sampleO[i].y + sampleB[i].y) * 0.5,
            (sampleO[i].z + sampleB[i].z) * 0.5,
          ),
      ];
      for (var t = 0; t <= 800; t += 100) {
        uncertainCapture.add([blended], t);
      }
      expect(uncertainCapture.letters, isEmpty);
      expect(uncertainCapture.retry, isTrue);
      expect(uncertainCapture.activePrediction, isNull);
    });

    test('AlphabetSignCapture detects dynamic J and Z finger trajectories', () {
      final capture = AlphabetSignCapture(alphabetClassifier);
      final sampleI = decodeSampleLandmarks(
          testSamples.firstWhere((s) => s['label'] == 'I'));
      final sampleD = decodeSampleLandmarks(
          testSamples.firstWhere((s) => s['label'] == 'D'));

      // Pinky downward-then-hook curve starting from 'I' -> dynamic 'J'.
      const jOffsets = <(double, double)>[
        (0.00, 0.00),
        (0.00, 0.07),
        (0.00, 0.14),
        (0.00, 0.21),
        (0.05, 0.26),
        (0.12, 0.25),
        (0.19, 0.21),
      ];
      for (var i = 0; i < jOffsets.length; i++) {
        final moved = [
          for (var k = 0; k < sampleI.length; k++)
            k == 20
                ? LandmarkPoint(
                    sampleI[k].x + jOffsets[i].$1,
                    sampleI[k].y + jOffsets[i].$2,
                    sampleI[k].z,
                  )
                : sampleI[k],
        ];
        capture.add([moved], i * 65);
      }
      expect(capture.letters, ['J']);
      expect(capture.isDynamicLetterActive, isTrue);

      // Clear and trace 3-segment zigzag starting from 'D' -> dynamic 'Z'.
      capture.add(const [], 1000);
      capture.add(const [], 1600);
      const zOffsets = <(double, double)>[
        (0.00, 0.00),
        (0.12, 0.00),
        (0.24, 0.00),
        (0.12, 0.10),
        (0.00, 0.20),
        (-0.12, 0.28),
        (0.02, 0.28),
        (0.18, 0.28),
      ];
      for (var i = 0; i < zOffsets.length; i++) {
        final moved = [
          for (var k = 0; k < sampleD.length; k++)
            k == 8
                ? LandmarkPoint(
                    sampleD[k].x + zOffsets[i].$1,
                    sampleD[k].y + zOffsets[i].$2,
                    sampleD[k].z,
                  )
                : sampleD[k],
        ];
        capture.add([moved], 1700 + i * 65);
      }
      expect(capture.letters, ['J', 'Z']);
      expect(capture.isDynamicLetterActive, isTrue);
    });
  });

  group('FSL-105 v4 dynamic sign recognition', () {
    final fslClassifier = SignClassifier.decode(
        File('assets/ml/fsl105_signs.json').readAsStringSync());
    final fslLabelsMeta = (jsonDecode(
            File('assets/ml/fsl105_labels.json').readAsStringSync()) as List)
        .cast<Map<String, dynamic>>();
    final fslSamples = (jsonDecode(
                File('fsl105_keypoints/test_samples.json').readAsStringSync())
            as List)
        .cast<Map<String, dynamic>>();

    List<BodySignFrame> buildBodyFrames(
      Map<String, dynamic> sample, {
      int rotation = 0,
      bool mirror = false,
      double? overrideAspect,
    }) {
      final srcAspect = (sample['aspect'] as num).toDouble();
      final aspect = overrideAspect ?? srcAspect;
      final frames = (sample['frames'] as List).cast<Map<String, dynamic>>();
      final out = <BodySignFrame>[];

      double remapX(double ux) {
        var x = mirror ? 1.0 - ux : ux;
        if (overrideAspect != null) {
          x = 0.5 + (x - 0.5) * srcAspect / overrideAspect;
        }
        return x;
      }

      LandmarkPoint toSensorHand(double ux, double uy, double uz) {
        final x = remapX(ux);
        final y = uy;
        final z = overrideAspect != null ? uz * srcAspect / aspect : uz;
        return switch (rotation) {
          0 => LandmarkPoint(x, y, z),
          90 => LandmarkPoint(y, 1.0 - x, z),
          180 => LandmarkPoint(1.0 - x, 1.0 - y, z),
          270 => LandmarkPoint(1.0 - y, x, z),
          _ => throw ArgumentError('Unsupported rotation: $rotation'),
        };
      }

      for (final f in frames) {
        final rawPose = (f['pose'] as List).cast<List>();
        final pose = [
          for (final p in rawPose)
            (p[0] as num) == 0 && (p[1] as num) == 0
                ? const LandmarkPoint(0, 0, 0)
                : LandmarkPoint(
                    remapX((p[0] as num).toDouble()),
                    (p[1] as num).toDouble(),
                    0,
                  ),
        ];

        final hands = <List<LandmarkPoint>>[];
        for (final key in const ['hand0', 'hand1']) {
          final pts = (f[key] as List).cast<List>();
          final nonzero = pts.any((p) =>
              (p[0] as num) != 0 || (p[1] as num) != 0 || (p[2] as num) != 0);
          if (!nonzero) continue;
          hands.add([
            for (final p in pts)
              toSensorHand(
                (p[0] as num).toDouble(),
                (p[1] as num).toDouble(),
                (p[2] as num).toDouble(),
              ),
          ]);
        }
        out.add(LandmarkFrame(
          hands,
          rotation: rotation,
          aspect: aspect,
          pose: pose,
        ).modelBody);
      }
      return out;
    }

    test(
        'fsl105 model and labels exclude GOOD EVENING (id 2) and cover 104 classes',
        () {
      expect(fslClassifier.version, 4);
      expect(fslClassifier.steps, 16);
      expect(fslClassifier.features, 322);
      expect(fslClassifier.hidden, 128);
      expect(fslClassifier.labels.length, 104);
      expect(fslClassifier.labelIds, isNotNull);
      expect(fslClassifier.labelIds!.length, 104);
      expect(fslClassifier.labelIds, contains(0));
      expect(fslClassifier.labelIds, contains(1));
      expect(fslClassifier.labelIds, isNot(contains(2)));
      expect(fslClassifier.labelIds, contains(104));
      expect(fslClassifier.labels, contains('GOOD_AFTERNOON'));
      expect(fslClassifier.labels, isNot(contains('GOOD_EVENING')));
      expect(fslClassifier.labels,
          equals([for (final r in fslLabelsMeta) r['key'] as String]));
    });

    test('all 104 FSL-105 labels localize to non-empty English and Filipino text',
        () {
      const en = AppLocalizations(Locale('en'));
      const fil = AppLocalizations(Locale('fil'));
      for (final label in fslClassifier.labels) {
        final enText = en.signLabel(label);
        final filText = fil.signLabel(label);
        expect(enText, isNotEmpty, reason: 'Missing English text for $label');
        expect(filText, isNotEmpty, reason: 'Missing Filipino text for $label');
        expect(enText.startsWith('sign_'), isFalse,
            reason: 'Unresolved English key for $label');
        expect(filText.startsWith('sign_'), isFalse,
            reason: 'Unresolved Filipino key for $label');
      }
      expect(fil.signLabel('IM_FINE'), 'Mabuti naman ako');
    });

    test('Dart v4 predictions match Python reference on held-out Signer 4 clips',
        () {
      final focusGreetings = <String>{
        'GOOD_MORNING',
        'GOOD_AFTERNOON',
        'HOW_ARE_YOU',
        'IM_FINE',
        'NICE_TO_MEET_YOU',
        'THANK_YOU',
        'SEE_YOU_TOMORROW',
      };
      final checkedGreetings = <String>{};

      for (final sample in fslSamples) {
        final expectedPred = sample['expected_pred'] as String;
        final expectedScore = (sample['expected_score'] as num).toDouble();
        final expectedMotion = (sample['expected_motion'] as num).toDouble();
        final floor = fslClassifier.motionFloors[expectedPred];
        final frames = buildBodyFrames(sample);
        final prediction = fslClassifier.predict(frames);

        if (floor != null && expectedMotion < floor) {
          expect(prediction, isNull,
              reason: 'Expected motion-floor rejection for ${sample['label']}');
          continue;
        }
        expect(prediction, isNotNull,
            reason: 'Missing prediction for ${sample['label']}');
        expect(prediction!.label, expectedPred,
            reason: 'Label mismatch on ${sample['label']}');
        expect(prediction.score, closeTo(expectedScore, 1e-3),
            reason: 'Score mismatch on ${sample['label']}');

        if (focusGreetings.contains(sample['label'])) {
          expect(prediction.label, sample['label']);
          expect(fslClassifier.accepts(prediction), isTrue,
              reason: 'Expected accepted score for ${sample['label']}');
          checkedGreetings.add(sample['label'] as String);
        }
      }
      expect(checkedGreetings, equals(focusGreetings));
    });

    test(
        'predictions are invariant to mirror flip and 90/270 camera sensor rotations',
        () {
      const greetingTargets = [
        'GOOD_MORNING',
        'GOOD_AFTERNOON',
        'HOW_ARE_YOU',
        'IM_FINE',
        'NICE_TO_MEET_YOU',
      ];
      for (final key in greetingTargets) {
        final sample = fslSamples.firstWhere((s) => s['label'] == key);
        final expectedScore = (sample['expected_score'] as num).toDouble();

        final mirrored = fslClassifier.predict(
          buildBodyFrames(sample, mirror: true),
        );
        expect(mirrored, isNotNull);
        expect(mirrored!.label, key);
        expect(mirrored.score, closeTo(expectedScore, 1e-4));

        for (final rot in const [90, 270]) {
          final rotated = fslClassifier.predict(
            buildBodyFrames(sample, rotation: rot, overrideAspect: 3 / 4),
          );
          expect(rotated, isNotNull);
          expect(rotated!.label, key);
          expect(rotated.score, closeTo(expectedScore, 1e-4));
        }
      }
    });

    test(
        'frozen posture is rejected, SignCapture trims steady end posture, and fallback works without pose',
        () {
      final morningSample =
          fslSamples.firstWhere((s) => s['label'] == 'GOOD_MORNING');
      final morningFrames = buildBodyFrames(morningSample);
      final midFrame = morningFrames[morningFrames.length ~/ 2];
      final frozen = List<BodySignFrame>.filled(16, midFrame);
      expect(fslClassifier.predict(frozen), isNull);

      final combined = fslClassifier.withFallback(classifier);
      final dynamicPred = combined.predict(morningFrames);
      expect(dynamicPred, isNotNull);
      expect(dynamicPred!.label, 'GOOD_MORNING');
      expect(combined.accepts(dynamicPred), isTrue);

      final helloFiles =
          Directory('coordinates/HELLO').listSync().whereType<File>().toList();
      final helloFrames = _readClip(
        File('coordinates/HELLO/HELLO_${helloFiles.length}.npy'),
      );
      expect(fslClassifier.predict(helloFrames), isNull);
      final fallbackPred = combined.predict(helloFrames);
      expect(fallbackPred, isNotNull);
      expect(fallbackPred!.label, 'HELLO');
      expect(combined.accepts(fallbackPred), isTrue);

      final capture = SignCapture(combined);
      var t = 0;
      for (final frame in morningFrames) {
        capture.add(frame, t);
        t += 80;
      }
      // Hold final ending posture steady to trigger anchor-based early completion.
      for (var k = 0; k < 12 && capture.result == null; k++) {
        capture.add(morningFrames.last, t);
        t += 90;
      }
      expect(capture.result, isNotNull);
      expect(capture.result!.label, 'GOOD_MORNING');
    });

    test(
        'v4 handles previously difficult signs, speed variations, duplicate suppression, and stray leading blips',
        () {
      const difficultTop1 = [
        'CRAB',
        'TUESDAY',
        'SON',
        'YOURE_WELCOME',
        'FATHER',
        'UNCLE',
        'BLIND',
        'TODAY',
        'WOMAN',
        'HARD_OF_HEARING',
        'FOUR',
      ];
      const difficultAccepted = [
        'TUESDAY',
        'SON',
        'YOURE_WELCOME',
        'FATHER',
        'BLIND',
        'TODAY',
        'WOMAN',
        'FOUR',
      ];
      for (final label in difficultTop1) {
        final sample = fslSamples.firstWhere((s) => s['label'] == label);
        final pred = fslClassifier.predict(buildBodyFrames(sample));
        expect(pred, isNotNull, reason: 'Expected prediction for $label');
        expect(pred!.label, label, reason: 'Top-1 mismatch for $label');
        if (difficultAccepted.contains(label)) {
          expect(fslClassifier.accepts(pred), isTrue,
              reason: 'Expected accepts=true for $label');
        }
      }

      List<BodySignFrame> resampleSpeed(
          List<BodySignFrame> src, int targetLen) {
        return [
          for (var i = 0; i < targetLen; i++)
            src[((i * (src.length - 1)) / (targetLen - 1)).round()],
        ];
      }

      for (final label in const ['GOOD_MORNING', 'TUESDAY', 'BLIND']) {
        final sample = fslSamples.firstWhere((s) => s['label'] == label);
        final frames = buildBodyFrames(sample);
        final fast = fslClassifier.predict(
          resampleSpeed(frames, math.max(8, (frames.length * 0.7).round())),
        );
        final slow = fslClassifier.predict(
          resampleSpeed(frames, (frames.length * 1.5).round()),
        );
        expect(fast?.label, label, reason: '0.7x speed failed for $label');
        expect(slow?.label, label, reason: '1.5x speed failed for $label');
        expect(fslClassifier.accepts(fast!), isTrue);
        expect(fslClassifier.accepts(slow!), isTrue);
      }

      // Duplicate phantom second-hand suppression & stray leading blip isolation.
      final morningFrames = buildBodyFrames(
        fslSamples.firstWhere((s) => s['label'] == 'GOOD_MORNING'),
      );
      final dupHandFrames = [
        for (final f in morningFrames)
          if (f.hands.length == 1)
            BodySignFrame(
              [
                f.hands.first,
                [
                  for (final p in f.hands.first)
                    (x: p.x + 0.01, y: p.y + 0.01),
                ],
              ],
              f.pose,
              [f.depth.first, f.depth.first],
            )
          else
            f,
      ];
      final dupPred = fslClassifier.predict(dupHandFrames);
      expect(dupPred?.label, 'GOOD_MORNING');
      expect(fslClassifier.accepts(dupPred!), isTrue);

      final strayPrefix = <BodySignFrame>[
        morningFrames.first,
        for (var i = 0; i < 12; i++)
          BodySignFrame(const [], morningFrames.first.pose, const []),
        ...morningFrames,
      ];
      final blipPred = fslClassifier.predict(strayPrefix);
      expect(blipPred?.label, 'GOOD_MORNING');
      expect(fslClassifier.accepts(blipPred!), isTrue);
    });
  });

  group('unified continuous FSL recognition session', () {
    final alphaConfig = jsonDecode(
      File('assets/ml/alphabet_preprocessing.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    final alphaLabels = List<String>.from(alphaConfig['labels'] as List);
    final alphaThreshold = (alphaConfig['threshold'] as num).toDouble();
    final alphaSamples = List<Map<String, dynamic>>.from(
      alphaConfig['testSamples'] as List,
    );
    final alphaClassifier = AlphabetClassifier.fromTfliteBytes(
      File('assets/ml/alphabet_classifier.tflite').readAsBytesSync(),
      labels: alphaLabels,
      threshold: alphaThreshold,
    );
    final txModel = SignClassifier.decode(
      File('assets/ml/transaction_signs.json').readAsStringSync(),
    );
    final fslModel = SignClassifier.decode(
      File('assets/ml/fsl105_signs.json').readAsStringSync(),
    );
    final combinedDynamic = fslModel.withFallback(txModel);
    final fslSamples = (jsonDecode(
      File('fsl105_keypoints/test_samples.json').readAsStringSync(),
    ) as List)
        .cast<Map<String, dynamic>>();

    List<LandmarkPoint> decodeAlphaHand(String letter) {
      final sample = alphaSamples.firstWhere((s) => s['label'] == letter);
      final raw = sample['landmarks'] as List;
      return [
        for (final p in raw)
          LandmarkPoint(
            (p[0] as num).toDouble(),
            (p[1] as num).toDouble(),
            (p[2] as num).toDouble(),
          ),
      ];
    }

    LandmarkFrame toAlphaFrame(
      List<LandmarkPoint> uprightHand, {
      double dx = 0.0,
    }) {
      const aspect = 3 / 4;
      const scale = 0.5;
      final points = [
        for (final p in uprightHand)
          LandmarkPoint(
            ((p.x + dx) * scale) / aspect + 0.1,
            p.y * scale + 0.15,
            p.z * scale / aspect,
          ),
      ];
      return LandmarkFrame([points], rotation: 0, aspect: aspect);
    }

    List<LandmarkFrame> buildDynamicLandmarkFrames(
      Map<String, dynamic> sample,
    ) {
      final aspect = (sample['aspect'] as num).toDouble();
      final frames = (sample['frames'] as List).cast<Map<String, dynamic>>();
      final out = <LandmarkFrame>[];
      for (final f in frames) {
        final rawPose = (f['pose'] as List).cast<List>();
        final pose = [
          for (final p in rawPose)
            (p[0] as num) == 0 && (p[1] as num) == 0
                ? const LandmarkPoint(0, 0, 0)
                : LandmarkPoint(
                    (p[0] as num).toDouble(),
                    (p[1] as num).toDouble(),
                    0,
                  ),
        ];
        final hands = <List<LandmarkPoint>>[];
        for (final key in const ['hand0', 'hand1']) {
          final pts = (f[key] as List).cast<List>();
          final nonzero = pts.any((p) =>
              (p[0] as num) != 0 || (p[1] as num) != 0 || (p[2] as num) != 0);
          if (!nonzero) continue;
          hands.add([
            for (final p in pts)
              LandmarkPoint(
                (p[0] as num).toDouble(),
                (p[1] as num).toDouble(),
                (p[2] as num).toDouble(),
              ),
          ]);
        }
        out.add(LandmarkFrame(hands, rotation: 0, aspect: aspect, pose: pose));
      }
      return out;
    }

    ContinuousSignSession createSession() => ContinuousSignSession(
          dynamicCapture: SignCapture(combinedDynamic),
          alphabetCapture: AlphabetSignCapture(alphaClassifier),
        );

    test(
        'fingerspells continuously, avoids hold duplicates, and supports bounce double letters & spacing',
        () {
      final session = createSession();
      const en = AppLocalizations(Locale('en'));
      var t = 0;

      void holdLetter(String letter, {int frames = 6, double dx = 0.0}) {
        final hand = decodeAlphaHand(letter);
        for (var i = 0; i < frames; i++) {
          session.addFrame(toAlphaFrame(hand, dx: dx), t, useBody: true);
          t += 100;
        }
      }

      // Holding 'H' for >1.5 seconds must only emit one 'H' and not set retry.
      holdLetter('H', frames: 16);
      expect(session.formatText(en), 'H');
      expect(session.retry, isFalse);

      holdLetter('E');
      holdLetter('L');
      // Intentional positional bounce (0.60 palm scale) allows second 'L'.
      final handL = decodeAlphaHand('L');
      final palmL = math.sqrt(
        math.pow(handL[9].x - handL[0].x, 2) +
            math.pow(handL[9].y - handL[0].y, 2),
      );
      session.addFrame(
        toAlphaFrame(handL, dx: 0.60 * palmL),
        t,
        useBody: true,
      );
      t += 100;
      holdLetter('L');
      holdLetter('O');

      session.insertSpace();
      holdLetter('F');
      holdLetter('L');
      holdLetter('Y');

      expect(session.formatText(en), 'HELLO FLY');
      expect(session.tokens.length, 9);
      expect(session.dynamicCapture.retry, isFalse);
    });

    test(
        'recognizes consecutive dynamic FSL signs without restarting and suppresses trailing static artifacts',
        () {
      final session = createSession();
      const en = AppLocalizations(Locale('en'));
      const fil = AppLocalizations(Locale('fil'));
      final handB = decodeAlphaHand('B');
      var t = 0;

      for (final label in const ['GOOD_MORNING', 'HOW_ARE_YOU', 'IM_FINE']) {
        final sample = fslSamples.firstWhere((s) => s['label'] == label);
        final frames = buildDynamicLandmarkFrames(sample);
        for (final f in frames) {
          session.addFrame(f, t, useBody: true);
          t += 80;
        }
        // Hold ending posture so SignCapture finishes while hand stays visible.
        for (var k = 0; k < 12; k++) {
          session.addFrame(frames.last, t, useBody: true);
          t += 90;
        }
        // Briefly pause before next sign (clears waiting-for-release after 300 ms).
        t += 400;
        session.addFrame(
          LandmarkFrame(const [], rotation: 0, aspect: frames.first.aspect),
          t,
          useBody: true,
        );
        t += 100;
      }

      // Verify post-dynamic release suppression prevents a held ending 'B' hand
      // right after a single-hand dynamic sign from appending an extra 'B'.
      final morningSample =
          fslSamples.firstWhere((s) => s['label'] == 'GOOD_MORNING');
      final morningFrames = buildDynamicLandmarkFrames(morningSample);
      final beforeCount = session.tokens.length;
      for (final f in morningFrames) {
        session.addFrame(f, t, useBody: true);
        t += 80;
      }
      for (var k = 0; k < 12 && session.tokens.length == beforeCount; k++) {
        session.addFrame(morningFrames.last, t, useBody: true);
        t += 90;
      }
      // Still holding ending posture without releasing -> alphabet suppressed.
      for (var k = 0; k < 5; k++) {
        session.addFrame(morningFrames.last, t, useBody: true);
        t += 100;
      }
      expect(
        session.tokens.where((tok) => tok.kind == RecognizedTokenKind.letter),
        isEmpty,
      );
      // Now drop out for >= 300 ms and sign static 'B' cleanly.
      t += 400;
      session.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 100;
      for (var k = 0; k < 6; k++) {
        session.addFrame(toAlphaFrame(handB), t, useBody: true);
        t += 100;
      }
      expect(session.tokens.last.kind, RecognizedTokenKind.letter);
      expect(session.tokens.last.label, 'B');
      session.undo();
      session.undo();

      expect(
        [for (final tok in session.tokens) tok.label],
        const ['GOOD_MORNING', 'HOW_ARE_YOU', 'IM_FINE'],
      );
      expect(
        session.formatText(en),
        "Good morning How are you? I'm fine",
      );
      expect(
        session.formatText(fil),
        'Magandang umaga Kumusta ka? Mabuti naman ako',
      );
    });

    test('supports mixed dynamic and static tokens, undo, space, clear, and resetSegment',
        () {
      final session = createSession();
      const en = AppLocalizations(Locale('en'));
      final handA = decodeAlphaHand('A');
      final handC = decodeAlphaHand('C');
      final morningFrames = buildDynamicLandmarkFrames(
        fslSamples.firstWhere((s) => s['label'] == 'GOOD_MORNING'),
      );
      var t = 0;

      for (var k = 0; k < 6; k++) {
        session.addFrame(toAlphaFrame(handA), t, useBody: true);
        t += 100;
      }
      for (var k = 0; k < 6; k++) {
        session.addFrame(toAlphaFrame(handC), t, useBody: true);
        t += 100;
      }
      // Pause to separate static letters from the next dynamic sign cleanly.
      t += 500;
      session.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 100;
      for (final f in morningFrames) {
        session.addFrame(f, t, useBody: true);
        t += 80;
      }
      for (var k = 0; k < 12; k++) {
        session.addFrame(morningFrames.last, t, useBody: true);
        t += 90;
      }
      t += 500;
      session.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 100;
      for (var k = 0; k < 6; k++) {
        session.addFrame(toAlphaFrame(handA), t, useBody: true);
        t += 100;
      }

      expect(session.formatText(en), 'AC Good morning A');
      session.resetSegment();
      expect(session.formatText(en), 'AC Good morning A');

      session.undo();
      expect(session.formatText(en), 'AC Good morning');
      session.undo();
      expect(session.formatText(en), 'AC');
      session.insertSpace();
      expect(session.formatText(en), 'AC ');
      session.clear();
      expect(session.hasTokens, isFalse);
      expect(session.formatText(en), isEmpty);
    });

    test(
        'distinguishes overlapping static handshapes from dynamic gestures and keeps temporary candidates separate',
        () {
      const en = AppLocalizations(Locale('en'));
      final handB = decodeAlphaHand('B');

      List<LandmarkFrame> singleHandClipFrames(String label) {
        final raw = buildDynamicLandmarkFrames(
          fslSamples.firstWhere((s) => s['label'] == label),
        );
        return [
          for (final f in raw)
            if (f.hands.isNotEmpty)
              LandmarkFrame(
                [
                  f.hands.reduce((a, b) => a.first.y <= b.first.y ? a : b),
                ],
                rotation: f.rotation,
                aspect: f.aspect,
                pose: f.pose,
                poseError: f.poseError,
                poseSampled: f.poseSampled,
              ),
        ];
      }

      // 1. Single-frame and brief 3-frame (200 ms) static handshapes expose only
      // temporaryPrediction without committing to confirmed output.
      final briefSession = createSession();
      var t = 0;
      for (var k = 0; k < 3; k++) {
        briefSession.addFrame(toAlphaFrame(handB), t, useBody: true);
        expect(briefSession.temporaryPrediction?.label, 'B');
        expect(briefSession.tokens, isEmpty);
        expect(briefSession.formatText(en), isEmpty);
        t += 100;
      }

      // Holding one extra steady frame (total 300 ms / 4 frames) confirms 'B'.
      briefSession.addFrame(toAlphaFrame(handB), t, useBody: true);
      expect(briefSession.formatText(en), 'B');
      expect(briefSession.temporaryPrediction, isNull);

      // 2. Starting with a brief overlapping 'B' handshape (200 ms and 400 ms
      // preparation holds) before sweeping into 1-hand GOOD_MORNING commits
      // GOOD_MORNING without leaking a leading 'B'.
      for (final prepFrames in const [3, 5]) {
        final session = createSession();
        t = 0;
        for (var k = 0; k < prepFrames; k++) {
          session.addFrame(toAlphaFrame(handB), t, useBody: true);
          t += 100;
        }
        final morning1h = singleHandClipFrames('GOOD_MORNING');
        for (final f in morning1h) {
          session.addFrame(f, t, useBody: true);
          t += 80;
        }
        for (var k = 0; k < 12; k++) {
          session.addFrame(morning1h.last, t, useBody: true);
          t += 90;
        }
        expect(
          session.formatText(en),
          'Good morning',
          reason: 'prepFrames=$prepFrames should be superseded by GOOD_MORNING',
        );
        expect(
          session.tokens.where((tok) => tok.kind == RecognizedTokenKind.letter),
          isEmpty,
        );
      }

      // 3. Single-hand dynamic gestures with alphabet-like intermediate/ending
      // postures (GOOD_MORNING, HELLO, IM_FINE, YES, NO) emit only dynamic tokens.
      final oneHandSession = createSession();
      t = 0;
      const singleHandDynamicTargets = [
        'GOOD_MORNING',
        'HELLO',
        'IM_FINE',
        'YES',
        'NO',
      ];
      for (final label in singleHandDynamicTargets) {
        final clip = singleHandClipFrames(label);
        for (final f in clip) {
          oneHandSession.addFrame(f, t, useBody: true);
          t += 80;
        }
        for (var k = 0; k < 12; k++) {
          oneHandSession.addFrame(clip.last, t, useBody: true);
          t += 90;
        }
        t += 400;
        oneHandSession.addFrame(
          LandmarkFrame(const [], rotation: 0, aspect: clip.first.aspect),
          t,
          useBody: true,
        );
        t += 100;
      }
      expect(
        [for (final tok in oneHandSession.tokens) tok.label],
        equals(singleHandDynamicTargets),
      );
      expect(
        oneHandSession.tokens
            .where((tok) => tok.kind == RecognizedTokenKind.letter),
        isEmpty,
      );

      // 4. Unmatched sweeping movement passing through overlapping static 'B'
      // handshapes must not commit a bogus static 'B' to confirmed output.
      final sweepSession = createSession();
      t = 0;
      final palmB = math.sqrt(
        math.pow(handB[9].x - handB[0].x, 2) +
            math.pow(handB[9].y - handB[0].y, 2),
      );
      const offsets = [0.0, 0.45, 0.90, 1.35, 1.40, 1.42];
      for (final mult in offsets) {
        sweepSession.addFrame(
          toAlphaFrame(handB, dx: mult * palmB),
          t,
          useBody: true,
        );
        t += 100;
      }
      t += 700;
      sweepSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      expect(sweepSession.tokens, isEmpty);
      expect(sweepSession.formatText(en), isEmpty);
    });

    test(
        'supports quick bounce unlatching, fast disappearance reset, rapid fingerspelling, and sub-5ms inference latency',
        () {
      // 1. Quick positional bounce unlatching (0.36 palm scale >= 0.32 unlatchWristShift)
      //    and fast disappearance clear (260 ms).
      final capture = AlphabetSignCapture(alphaClassifier);
      final handO = decodeAlphaHand('O');
      final palmO = math.sqrt(
        math.pow(handO[9].x - handO[0].x, 2) +
            math.pow(handO[9].y - handO[0].y, 2),
      );
      var t = 0;
      for (var i = 0; i < 6; i++) {
        capture.add(toAlphaFrame(handO).modelHands3D, t);
        t += 100;
      }
      expect(capture.letters, equals(['O']));
      expect(capture.activePrediction?.label, 'O');

      // Single quick bounce at 0.36 * palmScale unlatches 'O' without long reset.
      capture.add(toAlphaFrame(handO, dx: 0.36 * palmO).modelHands3D, t);
      t += 100;
      for (var i = 0; i < 6; i++) {
        capture.add(toAlphaFrame(handO).modelHands3D, t);
        t += 100;
      }
      expect(capture.letters, equals(['O', 'O']));

      // Empty frames >= 260 ms clear activePrediction quickly while keeping confirmed letters.
      capture.add(const [], t);
      t += 280;
      capture.add(const [], t);
      expect(capture.activePrediction, isNull);
      expect(capture.letters, equals(['O', 'O']));

      // 2. Rapid follow-up fingerspelling in ContinuousSignSession after first commit.
      final session = createSession();
      const en = AppLocalizations(Locale('en'));
      final handH = decodeAlphaHand('H');
      final handI = decodeAlphaHand('I');
      t = 0;
      for (var i = 0; i < 6; i++) {
        session.addFrame(toAlphaFrame(handH), t, useBody: true);
        t += 100;
      }
      for (var i = 0; i < 4; i++) {
        session.addFrame(toAlphaFrame(handI), t, useBody: true);
        t += 100;
      }
      expect(session.formatText(en), 'HI');

      // 3. Sub-5ms average latency benchmark for both v4 dynamic and static alphabet models.
      final morningSample =
          fslSamples.firstWhere((s) => s['label'] == 'GOOD_MORNING');
      final morningBody = [
        for (final lf in buildDynamicLandmarkFrames(morningSample)) lf.modelBody,
      ];
      for (var i = 0; i < 3; i++) {
        fslModel.predict(morningBody);
        alphaClassifier.predictHand(handO);
      }

      final dynSw = Stopwatch()..start();
      for (var i = 0; i < 50; i++) {
        fslModel.predict(morningBody);
      }
      dynSw.stop();

      final alphaSw = Stopwatch()..start();
      for (var i = 0; i < 50; i++) {
        alphaClassifier.predictHand(handO);
      }
      alphaSw.stop();

      final dynAvgUs = dynSw.elapsedMicroseconds / 50.0;
      final alphaAvgUs = alphaSw.elapsedMicroseconds / 50.0;
      expect(dynAvgUs, lessThan(5000.0),
          reason: 'v4 dynamic predict avg=${dynAvgUs.toStringAsFixed(1)} us');
      expect(alphaAvgUs, lessThan(5000.0),
          reason:
              'static alphabet predictHand avg=${alphaAvgUs.toStringAsFixed(1)} us');
    });

    test(
        'supports 30fps continuous fingerspelling without window cap deadlock and fast 300ms dynamic gaps',
        () {
      const en = AppLocalizations(Locale('en'));
      final session = createSession();
      final handH = decodeAlphaHand('H');
      final handI = decodeAlphaHand('I');
      final handO = decodeAlphaHand('O');

      // Simulate a 30 FPS (33 ms/frame) camera stream for H -> I -> O.
      var t = 0;
      for (final hand in [handH, handI, handO]) {
        for (var i = 0; i < 8; i++) {
          session.addFrame(toAlphaFrame(hand), t, useBody: true);
          t += 33;
        }
      }
      expect(session.formatText(en), 'HIO');

      // Fast 300 ms hands-down gap between consecutive dynamic signs commits both.
      final dynSession = createSession();
      final morning = buildDynamicLandmarkFrames(
        fslSamples.firstWhere((s) => s['label'] == 'GOOD_MORNING'),
      );
      final hello = buildDynamicLandmarkFrames(
        fslSamples.firstWhere((s) => s['label'] == 'HELLO'),
      );
      t = 0;
      for (final f in morning) {
        dynSession.addFrame(f, t, useBody: true);
        t += 80;
      }
      t += 300;
      dynSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 40;
      for (final f in hello) {
        dynSession.addFrame(f, t, useBody: true);
        t += 80;
      }
      t += 300;
      dynSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      expect(dynSession.formatText(en), 'Good morning Hello');
    });

    test(
        'prevents static/dynamic collisions across J/Z trajectories, rapid fingerspelling, lowering exits, and immediate post-gap signs',
        () {
      const en = AppLocalizations(Locale('en'));

      double palmOf(List<LandmarkPoint> h) => math.max(
            1e-4,
            math.sqrt(
              math.pow(h[9].x - h[0].x, 2) + math.pow(h[9].y - h[0].y, 2),
            ),
          );

      List<LandmarkPoint> centerHand(List<LandmarkPoint> h) {
        final k = 0.16 / palmOf(h);
        return [
          for (final p in h)
            LandmarkPoint(
              0.5 + (p.x - h[0].x) * k,
              0.5 + (p.y - h[0].y) * k,
              (p.z - h[0].z) * k,
            ),
        ];
      }

      List<LandmarkPoint> shiftHand(
        List<LandmarkPoint> h,
        double dx,
        double dy,
      ) =>
          [
            for (final p in h) LandmarkPoint(p.x + dx, p.y + dy, p.z),
          ];

      List<LandmarkPoint> lerpHand(
        List<LandmarkPoint> a,
        List<LandmarkPoint> b,
        double t,
      ) =>
          [
            for (var i = 0; i < 21; i++)
              LandmarkPoint(
                a[i].x + (b[i].x - a[i].x) * t,
                a[i].y + (b[i].y - a[i].y) * t,
                a[i].z + (b[i].z - a[i].z) * t,
              ),
          ];

      // 1. Stationary 'J'/'Z' handshapes must never confirm statically.
      final heldZCapture = AlphabetSignCapture(alphaClassifier);
      final heldZ = decodeAlphaHand('Z');
      var t = 0;
      for (var i = 0; i < 8; i++) {
        heldZCapture.add(toAlphaFrame(heldZ).modelHands3D, t);
        t += 66;
      }
      expect(heldZCapture.letters, isEmpty);
      expect(heldZCapture.activePrediction, isNull);

      final heldJCapture = AlphabetSignCapture(alphaClassifier);
      final heldJ = decodeAlphaHand('J');
      t = 0;
      for (var i = 0; i < 8; i++) {
        heldJCapture.add(toAlphaFrame(heldJ).modelHands3D, t);
        t += 66;
      }
      expect(heldJCapture.letters, isEmpty);
      expect(heldJCapture.activePrediction, isNull);

      // 2. 'I' lowered linearly must not phantom-'J', and 'D' swaying must not phantom-'Z'.
      final dropICapture = AlphabetSignCapture(alphaClassifier);
      final handI = decodeAlphaHand('I');
      final palmI = palmOf(handI);
      t = 0;
      for (var i = 0; i < 4; i++) {
        dropICapture.add(toAlphaFrame(handI).modelHands3D, t);
        t += 66;
      }
      for (var i = 1; i <= 5; i++) {
        dropICapture.add(
          toAlphaFrame(shiftHand(handI, 0.0, i * 0.10 * palmI)).modelHands3D,
          t,
        );
        t += 66;
      }
      expect(dropICapture.letters, equals(['I']));

      final swayDCapture = AlphabetSignCapture(alphaClassifier);
      final handD = decodeAlphaHand('D');
      final palmD = palmOf(handD);
      t = 0;
      for (final dx in const [
        0.0,
        0.0,
        0.0,
        0.0,
        0.08,
        0.16,
        0.08,
        0.0,
        -0.08,
        -0.16,
        -0.08,
        0.0,
      ]) {
        swayDCapture.add(
          toAlphaFrame(handD, dx: dx * palmD).modelHands3D,
          t,
        );
        t += 66;
      }
      expect(swayDCapture.letters, equals(['D']));

      // 3. Static 'X' followed by downward exit preserves 'X' without false dynamic output.
      final dropXSession = createSession();
      final handX = decodeAlphaHand('X');
      final palmX = palmOf(handX);
      t = 0;
      for (var i = 0; i < 6; i++) {
        dropXSession.addFrame(toAlphaFrame(handX), t, useBody: true);
        t += 66;
      }
      for (var i = 1; i <= 6; i++) {
        dropXSession.addFrame(
          toAlphaFrame(shiftHand(handX, 0.0, i * 0.14 * palmX)),
          t,
          useBody: true,
        );
        t += 66;
      }
      t += 400;
      dropXSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      expect(dropXSession.formatText(en), 'X');

      // 4. Rapid 66ms fingerspelling across transitions (BLUE, JAZZ, QUIZ).
      const jOffsets = [
        (0.0, 0.0),
        (0.0, 0.14),
        (0.0, 0.28),
        (0.0, 0.42),
        (-0.14, 0.46),
        (-0.28, 0.40),
        (-0.34, 0.30),
      ];
      const zOffsets = [
        (-0.20, 0.0),
        (0.0, 0.0),
        (0.22, 0.0),
        (0.04, 0.16),
        (-0.16, 0.32),
        (0.06, 0.32),
        (0.26, 0.32),
      ];
      for (final word in const ['BLUE', 'FIVE', 'JAZZ', 'QUIZ']) {
        final s = createSession();
        t = 0;
        String? prev;
        for (var i = 0; i < word.length; i++) {
          final ch = word[i];
          if (ch == 'J') {
            final baseI = centerHand(handI);
            final p = palmOf(baseI);
            for (final (dx, dy) in jOffsets) {
              s.addFrame(
                toAlphaFrame(shiftHand(baseI, dx * p, dy * p)),
                t,
                useBody: true,
              );
              t += 66;
            }
          } else if (ch == 'Z') {
            final baseD = centerHand(handD);
            final p = palmOf(baseD);
            for (final (dx, dy) in zOffsets) {
              s.addFrame(
                toAlphaFrame(shiftHand(baseD, dx * p, dy * p)),
                t,
                useBody: true,
              );
              t += 66;
            }
          } else {
            final nextH = centerHand(decodeAlphaHand(ch));
            if (prev != null && prev != 'J' && prev != 'Z') {
              final prevH = centerHand(decodeAlphaHand(prev));
              for (final frac in const [0.30, 0.65]) {
                s.addFrame(
                  toAlphaFrame(lerpHand(prevH, nextH, frac)),
                  t,
                  useBody: true,
                );
                t += 66;
              }
            }
            for (var k = 0; k < 6; k++) {
              s.addFrame(toAlphaFrame(nextH), t, useBody: true);
              t += 66;
            }
          }
          prev = ch;
        }
        t += 400;
        s.addFrame(const LandmarkFrame([]), t, useBody: true);
        expect(s.formatText(en), word, reason: 'Failed on $word');
      }

      // 5. Dynamic clip followed immediately after a >=280ms gap by the next
      // non-empty clip preserves both (THANK_YOU -> IM_FINE without empty intermediate frame).
      final directChain = createSession();
      t = 0;
      for (final label in const ['HELLO', 'THANK_YOU', 'IM_FINE']) {
        final clip = buildDynamicLandmarkFrames(
          fslSamples.firstWhere((e) => e['label'] == label),
        );
        for (final f in clip) {
          directChain.addFrame(f, t, useBody: true);
          t += 66;
        }
        t += 350;
      }
      directChain.addFrame(const LandmarkFrame([]), t, useBody: true);
      expect(
        [for (final tok in directChain.tokens) tok.label],
        const ['HELLO', 'THANK_YOU', 'IM_FINE'],
      );
    });

    test(
        'inserts a single automatic word space when lowering hands into neutral rest while blocking false pauses and repeated-letter spaces',
        () {
      const en = AppLocalizations(Locale('en'));
      const fil = AppLocalizations(Locale('fil'));

      double palmOf(List<LandmarkPoint> h) => math.max(
            1e-4,
            math.sqrt(
              math.pow(h[9].x - h[0].x, 2) + math.pow(h[9].y - h[0].y, 2),
            ),
          );

      List<LandmarkPoint> centerAt(
        List<LandmarkPoint> h, {
        double cx = 0.5,
        double cy = 0.42,
      }) {
        final k = 0.16 / palmOf(h);
        return [
          for (final p in h)
            LandmarkPoint(
              cx + (p.x - h[0].x) * k,
              cy + (p.y - h[0].y) * k,
              (p.z - h[0].z) * k,
            ),
        ];
      }

      List<LandmarkPoint> lerpHand(
        List<LandmarkPoint> a,
        List<LandmarkPoint> b,
        double t,
      ) =>
          [
            for (var i = 0; i < 21; i++)
              LandmarkPoint(
                a[i].x + (b[i].x - a[i].x) * t,
                a[i].y + (b[i].y - a[i].y) * t,
                a[i].z + (b[i].z - a[i].z) * t,
              ),
          ];

      int signWord(
        ContinuousSignSession session,
        String word,
        int tStart,
      ) {
        var t = tStart;
        String? prev;
        for (var i = 0; i < word.length; i++) {
          final ch = word[i];
          final nextH = centerAt(decodeAlphaHand(ch));
          if (prev != null && prev == ch) {
            // Intentional repeated letter via quick lateral bounce.
            final palm = palmOf(nextH);
            session.addFrame(
              toAlphaFrame(centerAt(decodeAlphaHand(ch), cx: 0.5 + 0.36 * palm)),
              t,
              useBody: true,
            );
            t += 66;
          } else if (prev != null) {
            final prevH = centerAt(decodeAlphaHand(prev));
            for (final frac in const [0.35, 0.70]) {
              session.addFrame(
                toAlphaFrame(lerpHand(prevH, nextH, frac)),
                t,
                useBody: true,
              );
              t += 66;
            }
          }
          for (var k = 0; k < 8; k++) {
            session.addFrame(toAlphaFrame(nextH), t, useBody: true);
            t += 66;
          }
          prev = ch;
        }
        return t;
      }

      // 1. R -> E -> A -> D -> lower hand out of frame -> B -> O -> O -> K == "READ BOOK".
      final exitSession = createSession();
      var t = signWord(exitSession, 'READ', 0);
      expect(exitSession.formatText(en), 'READ');

      final handD = centerAt(decodeAlphaHand('D'));
      final palmD = palmOf(handD);
      for (var step = 1; step <= 4; step++) {
        exitSession.addFrame(
          toAlphaFrame(centerAt(handD, cy: 0.42 + step * 0.25 * palmD)),
          t,
          useBody: true,
        );
        t += 55;
      }
      t += 220;
      exitSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 66;
      // Still no trailing space while hands remain down.
      expect(exitSession.formatText(en), 'READ');
      expect(exitSession.isWordBreakPending, isTrue);

      t = signWord(exitSession, 'BOOK', t);
      expect(exitSession.formatText(en), 'READ BOOK');
      expect(exitSession.formatText(fil), 'READ BOOK');
      expect(
        exitSession.tokens.where((tok) => tok.kind == RecognizedTokenKind.space),
        hasLength(1),
      );

      // 2. R -> E -> A -> D -> lower BOTH hands into in-frame neutral rest -> B -> O -> O -> K == "READ BOOK".
      final bothRestSession = createSession();
      t = signWord(bothRestSession, 'READ', 0);
      final leftRest = centerAt(decodeAlphaHand('B'), cx: 0.34, cy: 0.78);
      final rightRest = centerAt(decodeAlphaHand('A'), cx: 0.66, cy: 0.78);
      // Lower hands down then hold both resting low for ~200ms.
      bothRestSession.addFrame(
        toAlphaFrame(centerAt(handD, cy: 0.62)),
        t,
        useBody: true,
      );
      t += 60;
      for (var i = 0; i < 4; i++) {
        bothRestSession.addFrame(
          LandmarkFrame([leftRest, rightRest], aspect: 4 / 3),
          t,
          useBody: true,
        );
        t += 60;
      }
      // Stationary resting hands at torso/waist do not inject stray static letters.
      expect(bothRestSession.formatText(en), 'READ');

      t = signWord(bothRestSession, 'BOOK', t);
      expect(bothRestSession.formatText(en), 'READ BOOK');
      expect(
        bothRestSession.tokens
            .where((tok) => tok.kind == RecognizedTokenKind.space),
        hasLength(1),
      );

      // 3. Repeated letters (HELLO, BOOK), brief unlowered dropout, and dynamic FSL signs.
      final comboSession = createSession();
      t = signWord(comboSession, 'HELLO', 0);
      expect(comboSession.formatText(en), 'HELLO');

      // Brief tracking dropout without lowering must NOT insert a space.
      t += 240;
      comboSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 66;
      expect(comboSession.isWordBreakPending, isFalse);

      // Lower hand twice in neutral rest before next word -> only ONE space inserted.
      for (var step = 1; step <= 4; step++) {
        comboSession.addFrame(
          toAlphaFrame(centerAt(decodeAlphaHand('O'), cy: 0.42 + step * 0.25 * palmD)),
          t,
          useBody: true,
        );
        t += 55;
      }
      t += 220;
      comboSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 120;
      comboSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 66;

      t = signWord(comboSession, 'BOOK', t);
      expect(comboSession.formatText(en), 'HELLO BOOK');

      // Follow with lower + dynamic sign -> clean spacing without duplicate space tokens.
      for (var step = 1; step <= 4; step++) {
        comboSession.addFrame(
          toAlphaFrame(centerAt(decodeAlphaHand('K'), cy: 0.42 + step * 0.25 * palmD)),
          t,
          useBody: true,
        );
        t += 55;
      }
      t += 240;
      comboSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      t += 66;

      final morning = buildDynamicLandmarkFrames(
        fslSamples.firstWhere((e) => e['label'] == 'GOOD_MORNING'),
      );
      for (final f in morning) {
        comboSession.addFrame(f, t, useBody: true);
        t += 70;
      }
      t += 320;
      comboSession.addFrame(const LandmarkFrame([]), t, useBody: true);
      expect(comboSession.formatText(en), 'HELLO BOOK Good morning');
      expect(
        comboSession.tokens
            .where((tok) => tok.kind == RecognizedTokenKind.space),
        hasLength(1),
      );
    });
  });
}
