import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum TrackingOverlayStyle {
  contour,
  skeleton,
}

class AppPreferences {
  AppPreferences._(this._values, [this._storage]);

  final Map<String, Object?> _values;
  final SharedPreferences? _storage;

  static Future<AppPreferences> create() async {
    final storage = await SharedPreferences.getInstance();
    return AppPreferences._(<String, Object?>{
      'locale': storage.getString('locale') ?? 'en',
      'hasOnboarded': storage.getBool('hasOnboarded') ?? false,
      'highContrast': storage.getBool('highContrast') ?? false,
      'reducedMotion': storage.getBool('reducedMotion') ?? false,
      'textScale': storage.getDouble('textScale') ?? 1.0,
      'overlayStyle':
          storage.getString('overlayStyle') ?? TrackingOverlayStyle.contour.name,
    }, storage);
  }

  /// An isolated store for widget tests.
  @visibleForTesting
  static AppPreferences inMemory() => AppPreferences._(<String, Object?>{
        'locale': 'en',
        'hasOnboarded': false,
        'highContrast': false,
        'reducedMotion': false,
        'textScale': 1.0,
        'overlayStyle': TrackingOverlayStyle.contour.name,
      });

  Locale get locale => Locale((_values['locale'] as String?) ?? 'en');
  bool get hasOnboarded => (_values['hasOnboarded'] as bool?) ?? false;
  bool get highContrast => (_values['highContrast'] as bool?) ?? false;
  bool get reducedMotion => (_values['reducedMotion'] as bool?) ?? false;
  double get textScale => (_values['textScale'] as double?) ?? 1.0;
  TrackingOverlayStyle get overlayStyle {
    final stored = _values['overlayStyle'] as String?;
    return stored == TrackingOverlayStyle.skeleton.name
        ? TrackingOverlayStyle.skeleton
        : TrackingOverlayStyle.contour;
  }

  bool get useSkeletonOverlay => overlayStyle == TrackingOverlayStyle.skeleton;

  Future<void> setLocale(Locale locale) async {
    _values['locale'] = locale.languageCode;
    await _storage?.setString('locale', locale.languageCode);
  }

  Future<void> setHasOnboarded(bool value) async {
    _values['hasOnboarded'] = value;
    await _storage?.setBool('hasOnboarded', value);
  }

  Future<void> setHighContrast(bool value) async {
    _values['highContrast'] = value;
    await _storage?.setBool('highContrast', value);
  }

  Future<void> setReducedMotion(bool value) async {
    _values['reducedMotion'] = value;
    await _storage?.setBool('reducedMotion', value);
  }

  Future<void> setTextScale(double value) async {
    _values['textScale'] = value;
    await _storage?.setDouble('textScale', value);
  }

  Future<void> setOverlayStyle(TrackingOverlayStyle style) async {
    _values['overlayStyle'] = style.name;
    await _storage?.setString('overlayStyle', style.name);
  }
}
