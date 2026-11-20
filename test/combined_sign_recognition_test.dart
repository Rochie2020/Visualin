import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:visualin/l10n/app_localizations.dart';
import 'package:visualin/services/alphabet_classifier.dart';
import 'package:visualin/services/alphabet_sign_capture.dart';
import 'package:visualin/services/continuous_sign_session.dart';
import 'package:visualin/services/landmark_service.dart';
import 'package:visualin/services/sign_capture.dart';
import 'package:visualin/services/sign_classifier.dart';

const _newTransactionLabels = <String>{
  'AGAIN',
  'CARD',
  'CASH',
  'COIN',
  'DISCOUNT',
  'HOW_MANY',
  'HOW_MUCH',
  'PLEASE',
  'PROBLEM',
  'RECEIPT',
  'WAIT',
  'WELCOME',
};

List<BodySignFrame> _decodeSampleBodyFrames(
  Map<String, dynamic> sample, {
  bool mirror = false,
}) {
  final aspect = (sample['aspect'] as num).toDouble();
  final rawFrames = (sample['frames'] as List).cast<Map<String, dynamic>>();
  return [
    for (final f in rawFrames) _decodeSingleBodyFrame(f, aspect, mirror: mirror),
  ];
}

BodySignFrame _decodeSingleBodyFrame(
  Map<String, dynamic> f,
  double aspect, {
  bool mirror = false,
}) {
  final rawPose = (f['pose'] as List).cast<List>();
  final pose = <SignPoint>[
    for (final p in rawPose)
      (
        x: (mirror ? 1.0 - (p[0] as num).toDouble() : (p[0] as num).toDouble()) *
            aspect,
        y: (p[1] as num).toDouble(),
      ),
  ];

  final hands = <List<SignPoint>>[];
  final depth = <List<double>>[];
  for (final key in const ['hand0', 'hand1']) {
    final pts = (f[key] as List?)?.cast<List>();
    if (pts == null || pts.length != 21) continue;
    final nonzero = pts.any(
      (p) => (p[0] as num) != 0 || (p[1] as num) != 0 || (p[2] as num) != 0,
    );
    if (!nonzero) continue;
    hands.add([
      for (final p in pts)
        (
          x: (mirror
                  ? 1.0 - (p[0] as num).toDouble()
                  : (p[0] as num).toDouble()) *
              aspect,
          y: (p[1] as num).toDouble(),
        ),
    ]);
    depth.add([
      for (final p in pts) (p[2] as num).toDouble() * aspect,
    ]);
  }
  return BodySignFrame(hands, pose, depth);
}

List<LandmarkFrame> _decodeSampleLandmarkFrames(
  Map<String, dynamic> sample, {
  int rotation = 0,
}) {
  final aspect = (sample['aspect'] as num).toDouble();
  final rawFrames = (sample['frames'] as List).cast<Map<String, dynamic>>();
  final out = <LandmarkFrame>[];
  for (final f in rawFrames) {
    final rawPose = (f['pose'] as List).cast<List>();
    final pose = [
      for (final p in rawPose)
        LandmarkPoint((p[0] as num).toDouble(), (p[1] as num).toDouble(), 0),
    ];
    final hands = <List<LandmarkPoint>>[];
    for (final key in const ['hand0', 'hand1']) {
      final pts = (f[key] as List?)?.cast<List>();
      if (pts == null || pts.length != 21) continue;
      final nonzero = pts.any(
        (p) => (p[0] as num) != 0 || (p[1] as num) != 0 || (p[2] as num) != 0,
      );
      if (!nonzero) continue;
      hands.add([
        for (final p in pts)
          _uprightToSensorPoint(
            (p[0] as num).toDouble(),
            (p[1] as num).toDouble(),
            (p[2] as num).toDouble(),
            rotation,
          ),
      ]);
    }
    out.add(
      LandmarkFrame(
        hands,
        rotation: rotation,
        aspect: aspect,
        pose: pose,
      ),
    );
  }
  return out;
}

LandmarkPoint _uprightToSensorPoint(
  double x,
  double y,
  double z,
  int rotation,
) =>
    switch (rotation) {
      90 => LandmarkPoint(y, 1 - x, z),
      180 => LandmarkPoint(1 - x, 1 - y, z),
      270 => LandmarkPoint(1 - y, x, z),
      _ => LandmarkPoint(x, y, z),
    };

List<LandmarkPoint> _decodeAlphabetLandmarks(Map<String, dynamic> sample) {
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

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final combined = SignClassifier.decode(
    File('assets/ml/fsl_combined_signs.json').readAsStringSync(),
  );
  final fsl105 = SignClassifier.decode(
    File('assets/ml/fsl105_signs.json').readAsStringSync(),
  );
  final txFallback = SignClassifier.decode(
    File('assets/ml/transaction_signs.json').readAsStringSync(),
  );
  final combinedMeta = (jsonDecode(
    File('assets/ml/fsl_combined_labels.json').readAsStringSync(),
  ) as List)
      .cast<Map<String, dynamic>>();
  final txSamples = (jsonDecode(
    File('transaction_keypoints/test_samples.json').readAsStringSync(),
  ) as List)
      .cast<Map<String, dynamic>>();
  final fslSamples = (jsonDecode(
    File('fsl105_keypoints/test_samples.json').readAsStringSync(),
  ) as List)
      .cast<Map<String, dynamic>>();

  final alphaConfig = jsonDecode(
    File('assets/ml/alphabet_preprocessing.json').readAsStringSync(),
  ) as Map<String, dynamic>;
  final alphaClassifier = AlphabetClassifier.fromTfliteBytes(
    File('assets/ml/alphabet_classifier.tflite').readAsBytesSync(),
    labels: List<String>.from(alphaConfig['labels'] as List),
    threshold: (alphaConfig['threshold'] as num).toDouble(),
  );
  final alphaSamples =
      List<Map<String, dynamic>>.from(alphaConfig['testSamples'] as List);

  group('Combined 116-class FSL model & labels', () {
    test('preserves FSL-105 + 12 new transaction classes with zero duplicates',
        () {
      expect(combined.version, 4);
      expect(combined.features, 322);
      expect(combined.steps, 16);
      expect(combined.hidden, 128);
      expect(combined.threshold, closeTo(0.55, 1e-6));
      expect(combined.labels.length, 116);
      expect(combined.labels.toSet().length, 116);

      // Existing FSL-105 and hand-only fallback assets remain valid.
      expect(fsl105.version, 4);
      expect(fsl105.labels.length, 104);
      expect(txFallback.version, 1);
      expect(txFallback.features, 91);
      expect(txFallback.labels.length, 26);

      // Every FSL-105 label and every new transaction label is in combined.
      expect(fsl105.labels.every(combined.labels.contains), isTrue);
      expect(_newTransactionLabels.every(combined.labels.contains), isTrue);
      expect(txFallback.labels.every(combined.labels.contains), isTrue);

      // Metadata file matches model label indices 1:1.
      expect(combinedMeta.length, 116);
      for (var i = 0; i < combinedMeta.length; i++) {
        expect(combinedMeta[i]['model_index'], i);
        expect(combinedMeta[i]['key'], combined.labels[i]);
      }

      // Adaptive motion floors exist for every class and allow slow valid signs.
      for (final label in combined.labels) {
        final floor = combined.motionFloors[label];
        expect(floor, isNotNull, reason: 'Missing motionFloor for $label');
        expect(floor!, greaterThan(0.0));
      }
      expect(combined.motionFloors['FIVE']!, lessThan(0.025));
      expect(combined.motionFloors['NINE']!, lessThan(0.025));
    });

    test('all 116 labels localize distinctly in English and Filipino', () {
      const en = AppLocalizations(Locale('en'));
      const fil = AppLocalizations(Locale('fil'));

      final enValues = <String>{};
      final filValues = <String>{};
      for (final key in combined.labels) {
        final enText = en.signLabel(key);
        final filText = fil.signLabel(key);
        expect(enText, isNotEmpty, reason: 'Empty EN label for $key');
        expect(filText, isNotEmpty, reason: 'Empty FIL label for $key');
        expect(enText, isNot('sign_$key'),
            reason: 'Unlocalized EN key for $key');
        expect(filText, isNot('sign_$key'),
            reason: 'Unlocalized FIL key for $key');
        enValues.add(enText);
        filValues.add(filText);
      }

      expect(enValues.length, combined.labels.length);
      expect(filValues.length, combined.labels.length);

      expect(en.signLabel('PLEASE'), 'Please');
      expect(fil.signLabel('PLEASE'), 'Pakiusap');
      expect(en.signLabel('WELCOME'), 'Welcome');
      expect(fil.signLabel('WELCOME'), 'Maligayang pagdating');
      expect(en.signLabel('YOURE_WELCOME'), "You're welcome");
      expect(en.signLabel('RECEIPT'), 'Receipt');
      expect(fil.signLabel('RECEIPT'), 'Resibo');
    });
  });

  group('Python-to-Dart parity & invariance on unseen signers', () {
    test('exact parity on all 26 held-out transaction dataset classes', () {
      expect(txSamples.length, 26);
      for (final sample in txSamples) {
        final label = sample['label'] as String;
        final expectedPred = sample['expected_pred'] as String;
        final expectedScore = (sample['expected_score'] as num).toDouble();
        final frames = _decodeSampleBodyFrames(sample);

        final pred = combined.predict(frames);
        expect(pred, isNotNull, reason: 'Null prediction for $label');
        expect(pred!.label, expectedPred, reason: 'Mismatch on $label');
        expect(pred.label, label, reason: 'Wrong top-1 for $label');
        expect(pred.score, closeTo(expectedScore, 1e-4),
            reason: 'Score drift on $label');
        expect(combined.accepts(pred), isTrue,
            reason: 'Should accept $label');
      }
    });

    test('preserves recognition across held-out FSL-105 Signer-4 test samples',
        () {
      var correct = 0;
      var accepted = 0;
      for (final sample in fslSamples) {
        final trueLabel = sample['label'] as String;
        final frames = _decodeSampleBodyFrames(sample);
        final pred = combined.predict(frames);
        if (pred != null && pred.label == trueLabel) {
          correct++;
          if (combined.accepts(pred)) accepted++;
        }
      }
      expect(correct / fslSamples.length, greaterThanOrEqualTo(0.90),
          reason: 'FSL-105 accuracy dropped: $correct/${fslSamples.length}');
      expect(accepted / fslSamples.length, greaterThanOrEqualTo(0.85));
    });

    test('mirror flip and 90/270 sensor rotation invariance on new signs', () {
      const targetLabels = [
        'AGAIN',
        'CARD',
        'PLEASE',
        'RECEIPT',
        'WELCOME',
        'FIVE',
        'THANK_YOU',
      ];
      for (final label in targetLabels) {
        final sample = txSamples.firstWhere((s) => s['label'] == label);
        final baseFrames = _decodeSampleBodyFrames(sample);
        final basePred = combined.predict(baseFrames)!;

        // Mirror flip preserves label and exact probability.
        final mirroredFrames = _decodeSampleBodyFrames(sample, mirror: true);
        final mirrorPred = combined.predict(mirroredFrames)!;
        expect(mirrorPred.label, basePred.label,
            reason: 'Mirror changed label for $label');
        expect(mirrorPred.score, closeTo(basePred.score, 1e-4),
            reason: 'Mirror changed score for $label');

        // 90 and 270 degree sensor rotations through LandmarkFrame.modelBody.
        for (final rot in const [90, 270]) {
          final rotated = _decodeSampleLandmarkFrames(sample, rotation: rot)
              .map((lf) => lf.modelBody)
              .toList();
          final rotPred = combined.predict(rotated);
          expect(rotPred?.label, basePred.label,
              reason: 'Rotation $rot changed label for $label');
          expect(rotPred!.score, closeTo(basePred.score, 1e-4));
        }
      }
    });
  });

  group('Fallback routing & real-time integration', () {
    test('blocks frozen static posture leakage even with txFallback attached',
        () {
      final guarded = combined.withFallback(txFallback);
      for (final label in const ['PLEASE', 'RECEIPT', 'WAIT', 'FIVE']) {
        final sample = txSamples.firstWhere((s) => s['label'] == label);
        final bodyFrames = _decodeSampleBodyFrames(sample);
        final frozen = List<BodySignFrame>.filled(16, bodyFrames[20]);
        expect(combined.predict(frozen), isNull,
            reason: 'Combined model should reject frozen $label');
        expect(guarded.predict(frozen), isNull,
            reason: 'Fallback must not bypass motionFloors on frozen $label');
      }
    });

    test('falls back cleanly to rebuilt transaction_signs when pose is absent',
        () {
      final guarded = combined.withFallback(txFallback);
      for (final label in const ['CARD', 'PLEASE', 'RECEIPT', 'WELCOME']) {
        final sample = txSamples.firstWhere((s) => s['label'] == label);
        final bodyFrames = _decodeSampleBodyFrames(sample);
        final handOnlyFrames = <SignFrame>[
          for (final bf in bodyFrames) bf.hands,
        ];
        final capture = SignCapture(guarded);
        for (var i = 0; i < handOnlyFrames.length; i++) {
          capture.add(handOnlyFrames[i], i * 40);
        }
        capture.add([], handOnlyFrames.length * 40 + 700);
        expect(capture.labels, contains(label),
            reason: 'Pose-free fallback missed $label');
      }
    });

    test(
        'ContinuousSignSession combines new dynamic signs, FSL-105 signs, and alphabet',
        () {
      final guarded = combined.withFallback(txFallback);
      final session = ContinuousSignSession(
        dynamicCapture: SignCapture(guarded),
        alphabetCapture: AlphabetSignCapture(alphaClassifier),
      );

      var t = 0;
      void streamDynamicSample(Map<String, dynamic> sample) {
        final frames = _decodeSampleLandmarkFrames(sample);
        for (final frame in frames) {
          session.addFrame(frame, t, useBody: true);
          t += 40;
        }
        // Release window (> 650ms gap / empty frames).
        for (var i = 0; i < 4; i++) {
          session.addFrame(const LandmarkFrame([]), t, useBody: true);
          t += 220;
        }
      }

      void streamStaticLetter(String letter) {
        final sample = alphaSamples.firstWhere((s) => s['label'] == letter);
        final hand = _decodeAlphabetLandmarks(sample);
        final frame = LandmarkFrame([hand], aspect: 4 / 3);
        for (var i = 0; i < 14; i++) {
          session.addFrame(frame, t, useBody: true);
          t += 55;
        }
        for (var i = 0; i < 5; i++) {
          session.addFrame(const LandmarkFrame([]), t, useBody: true);
          t += 120;
        }
      }

      final pleaseSample = txSamples.firstWhere((s) => s['label'] == 'PLEASE');
      final morningSample =
          fslSamples.firstWhere((s) => s['label'] == 'GOOD_MORNING');
      final receiptSample = txSamples.firstWhere((s) => s['label'] == 'RECEIPT');

      streamDynamicSample(pleaseSample);
      streamStaticLetter('A');
      streamStaticLetter('B');
      streamDynamicSample(morningSample);
      streamDynamicSample(receiptSample);

      expect(
        session.tokens.map((tok) => (tok.kind, tok.label)).toList(),
        [
          (RecognizedTokenKind.dynamicSign, 'PLEASE'),
          (RecognizedTokenKind.letter, 'A'),
          (RecognizedTokenKind.letter, 'B'),
          (RecognizedTokenKind.dynamicSign, 'GOOD_MORNING'),
          (RecognizedTokenKind.dynamicSign, 'RECEIPT'),
        ],
      );

      expect(
        session.formatText(const AppLocalizations(Locale('en'))),
        'Please AB Good morning Receipt',
      );
      expect(
        session.formatText(const AppLocalizations(Locale('fil'))),
        'Pakiusap AB Magandang umaga Resibo',
      );
    });

    test('combined model inference completes within 5ms per prediction', () {
      final sample = txSamples.firstWhere((s) => s['label'] == 'WELCOME');
      final frames = _decodeSampleBodyFrames(sample);
      expect(combined.predict(frames), isNotNull);

      const iterations = 100;
      final sw = Stopwatch()..start();
      for (var i = 0; i < iterations; i++) {
        combined.predict(frames);
      }
      sw.stop();
      final usPerCall = sw.elapsedMicroseconds / iterations;
      expect(usPerCall, lessThan(5000));
    });
  });
}
