import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:visualin/services/landmark_service.dart';
import 'package:visualin/widgets/landmark_overlay.dart';
import 'package:visualin/widgets/signing_workspace.dart';

void main() {
  testWidgets('short independent hand and pose gaps survive, then expire',
      (tester) async {
    final overlay = LandmarkOverlay();
    const a = [LandmarkPoint(.1, .2, 0)], b = [LandmarkPoint(.8, .4, 0)];
    overlay.update(const LandmarkFrame([a, b], pose: a));
    overlay.update(const LandmarkFrame([b]));
    await tester.pump(const Duration(milliseconds: 150));
    expect(overlay.hands, [a, b]);
    expect(overlay.pose, a);
    overlay.update(const LandmarkFrame([b, a], pose: a));
    await tester.pump(const Duration(milliseconds: 180));
    expect(overlay.hands, [a, b]);
    expect(overlay.pose, a);
    overlay.update(const LandmarkFrame([a], poseSampled: false));
    await tester.pump(const Duration(milliseconds: 251));
    expect(overlay.hands, [a]);
    expect(overlay.pose, a);
    overlay.update(const LandmarkFrame([]));
    await tester.pump(const Duration(milliseconds: 251));
    expect(overlay.hands, isEmpty);
    expect(overlay.pose, isEmpty);
    overlay.update(const LandmarkFrame([b], pose: b));
    overlay.reset();
    expect(overlay.hands, isEmpty);
    expect(overlay.pose, isEmpty);
    overlay.dispose();
  });

  testWidgets('large text and scrolling controls preserve the camera viewport',
      (tester) async {
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(tester.view.resetPhysicalSize);
    for (final size in [
      const Size(320, 480),
      const Size(360, 640),
      const Size(640, 320)
    ]) {
      tester.view.physicalSize = size;
      await tester.pumpWidget(MaterialApp(
          home: MediaQuery(
              data: MediaQueryData(
                  size: size, textScaler: const TextScaler.linear(1.8)),
              child: Scaffold(
                  body: SigningWorkspace(
                      preview: const ColoredBox(
                          key: ValueKey('preview'), color: Colors.black),
                      controls: [
                    const Text('Translated text'),
                    Text(List.filled(75, 'Good morning').join(' ')),
                    SwitchListTile(
                        title: const Text('Upper-body tracking'),
                        value: true,
                        onChanged: (_) {})
                  ])))));
      final preview = find.byKey(const ValueKey('preview'));
      final before = tester.getRect(preview);
      expect(before.height, greaterThan(size.height * .55));
      expect(before.width, greaterThan(size.width * .55));
      await tester.scrollUntilVisible(find.byType(SwitchListTile), 160);
      expect(tester.getRect(preview), before);
      expect(tester.takeException(), isNull);
    }
  });
}
