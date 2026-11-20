import 'package:flutter/widgets.dart';

class AppLocalizations {
  const AppLocalizations(this.locale);

  final Locale locale;

  static const supportedLocales = <Locale>[Locale('en'), Locale('fil')];
  static const localizationsDelegates = <LocalizationsDelegate<dynamic>>[
    _AppLocalizationsDelegate(),
  ];

  static AppLocalizations of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations)!;
  }

  bool get isEnglish => locale.languageCode == 'en';
  String _t(String key) => _messages[locale.languageCode]![key]!;

  String get tagline => _t('tagline');
  String get appName => _t('appName');
  String get welcome => _t('welcome');
  String get skip => _t('skip');
  String get next => _t('next');
  String get getStarted => _t('getStarted');
  String get onboardingSignText => _t('onboardingSignText');
  String get onboardingSignTextBody => _t('onboardingSignTextBody');
  String get onboardingTextSign => _t('onboardingTextSign');
  String get onboardingTextSignBody => _t('onboardingTextSignBody');
  String get onboardingSpeech => _t('onboardingSpeech');
  String get onboardingSpeechBody => _t('onboardingSpeechBody');
  String get onboardingCameraTitle => _t('onboardingCameraTitle');
  String get onboardingCameraBody => _t('onboardingCameraBody');
  String get cameraPermissionTitle => _t('cameraPermissionTitle');
  String get cameraPermissionBody => _t('cameraPermissionBody');
  String get homePrompt => _t('homePrompt');
  String get fslToText => _t('fslToText');
  String get fslToTextBody => _t('fslToTextBody');
  String get textToFsl => _t('textToFsl');
  String get textToFslBody => _t('textToFslBody');
  String get noTranslations => _t('noTranslations');
  String get noTranslationsBody => _t('noTranslationsBody');
  String get recentTranslations => _t('recentTranslations');
  String get home => _t('home');
  String get history => _t('history');
  String get flipCamera => _t('flipCamera');
  String get fullScreen => _t('fullScreen');
  String get exitFullScreen => _t('exitFullScreen');
  String get skeletonOverlayMode => _t('skeletonOverlayMode');
  String get skeletonOverlayDescription => _t('skeletonOverlayDescription');
  String get switchToSkeletonOverlay => _t('switchToSkeletonOverlay');
  String get switchToContourOverlay => _t('switchToContourOverlay');
  String get copiedToClipboard => _t('copiedToClipboard');
  String get cameraUnavailable => _t('cameraUnavailable');
  String get settings => _t('settings');
  String get outputLanguage => _t('outputLanguage');
  String get outputLanguageBody => _t('outputLanguageBody');
  String get translation => _t('translation');
  String get signLanguage => _t('signLanguage');
  String get signLanguageValue => _t('signLanguageValue');
  String get signLanguageBody => _t('signLanguageBody');
  String get filipino => _t('filipino');
  String get english => _t('english');
  String get accessibility => _t('accessibility');
  String get language => _t('language');
  String get app => _t('app');
  String get settingsSubtitle => _t('settingsSubtitle');
  String get fontSize => _t('fontSize');
  String get textSizeDescription => _t('textSizeDescription');
  String get highContrast => _t('highContrast');
  String get highContrastDescription => _t('highContrastDescription');
  String get reducedMotion => _t('reducedMotion');
  String get reducedMotionDescription => _t('reducedMotionDescription');
  String get cameraPreferences => _t('cameraPreferences');
  String get cameraPreferencesDescription => _t('cameraPreferencesDescription');
  String get helpSupport => _t('helpSupport');
  String get helpSupportDescription => _t('helpSupportDescription');
  String get about => _t('about');
  String get aboutDescription => _t('aboutDescription');
  String get smallText => _t('smallText');
  String get defaultText => _t('defaultText');
  String get largeText => _t('largeText');
  String get settingsFeatureUnavailable => _t('settingsFeatureUnavailable');
  String onboardingProgress(int current, int total) =>
      _t('onboardingProgress')
          .replaceAll('{current}', '$current')
          .replaceAll('{total}', '$total');
  String get historyStub => _t('historyStub');
  String get textToFslStub => _t('textToFslStub');
  String get permissionDenied => _t('permissionDenied');
  String get allowCamera => _t('allowCamera');
  String get later => _t('later');
  String get openSettings => _t('openSettings');
  String get handDetected => _t('handDetected');
  String get noHandDetected => _t('noHandDetected');
  String get cameraGuidanceNoHand => _t('cameraGuidanceNoHand');
  String get cameraGuidanceHandDetected => _t('cameraGuidanceHandDetected');
  String get cameraFrameLabel => _t('cameraFrameLabel');
  String get translatedText => _t('translatedText');
  String get datasetCreditBody => _t('datasetCreditBody');
  String get datasetCredit => _t('datasetCredit');
  String get bodyMode => _t('bodyMode');
  String get bodyError => _t('bodyError');
  String get bodyHint => _t('bodyHint');
  String get continuousMode => _t('continuousMode');
  String get continuousHint => _t('continuousHint');
  String get continuousRetry => _t('continuousRetry');
  String get continuousError => _t('continuousError');
  String get undoSign => _t('undoSign');
  String get clearText => _t('clearText');
  String get space => _t('space');
  String get alphabetMode => _t('alphabetMode');
  String get alphabetHint => _t('alphabetHint');
  String get alphabetRetry => _t('alphabetRetry');
  String get alphabetError => _t('alphabetError');
  String signLabel(String label) => _t('sign_$label');
  String get signRetry => _t('signRetry');
  String get translationWaiting => _t('translationWaiting');
  String get confidenceUnavailable => _t('confidenceUnavailable');
  String get speak => _t('speak');
  String get copy => _t('copy');
  String get save => _t('save');
  String fslToOutputLanguage(String outputLanguage) =>
      _t('fslToOutputLanguage').replaceAll('{language}', outputLanguage);
  String get on => _t('on');
  String get off => _t('off');
  String get close => _t('close');

  static const _messages = <String, Map<String, String>>{
    'fil': {
      'datasetCreditBody': 'FSL-105 v2 (2023), Isaiah Jassen Tupal. Ginagamit ng pagsubok na ito ang mga landmark ng kamay mula sa mga piling video.\nhttps://doi.org/10.17632/48y2y99mb9.2\nCC BY 4.0 - https://creativecommons.org/licenses/by/4.0/',
      'datasetCredit': 'Mga pagkilala sa dataset na FSL-105',
      'continuousMode': 'Tuloy-tuloy na pagsesenyas (pagsubok)',
    'bodyMode': 'Pagsubaybay sa itaas na katawan (42 / 104 senyas)',
    'bodyHint': 'Ipakita sa kamera ang mukha, balikat, siko, at mga kamay.',
    'bodyError': 'Hindi gumagana ang pagsubaybay sa katawan. I-off ito para magamit ang pagsubaybay sa kamay.',
    'sign_IM_FINE': "Mabuti naman ako",
    'sign_NICE_TO_MEET_YOU': "Ikinagagalak kitang makilala",
    'sign_SEE_YOU_TOMORROW': "Kita tayo bukas",
    'sign_MONDAY': "Lunes",
    'sign_TUESDAY': "Martes",
    'sign_WEDNESDAY': "Miyerkules",
    'sign_THURSDAY': "Huwebes",
    'sign_FRIDAY': "Biyernes",
    'sign_SATURDAY': "Sabado",
    'sign_SUNDAY': "Linggo",
      'continuousHint': 'Pagkatapos ng bawat senyas, panatilihin ang huling posisyon nang isang segundo bago simulan ang susunod.',
      'continuousRetry': 'Hindi malinaw ang senyas. Ulitin ang salitang iyon.',
      'continuousError': 'Hindi nabuksan ang tuloy-tuloy na pagsesenyas. Subukang muli.',
      'undoSign': 'Bawiin ang huling senyas',
      'alphabetMode': 'Modo ng Alpabeto (FSL)',
      'alphabetHint': 'Panatilihing matatag ang senyas ng titik sa FSL o iguhit ang J/Z.',
      'alphabetRetry': 'Hindi malinaw ang titik. Hawakan nang matatag ang kamay at subukang muli.',
      'alphabetError': 'Hindi nabuksan ang modelo ng alpabeto. Subukang muli.',
      'sign_GOOD_MORNING': 'Magandang umaga',
      'sign_GOOD_AFTERNOON': 'Magandang hapon',
      'sign_HOW_ARE_YOU': 'Kumusta ka?',
      'sign_TODAY': 'Ngayon',
      'sign_TOMORROW': 'Bukas',
      'sign_AGAIN': 'Muli',
      'sign_CARD': 'Kard',
      'sign_CASH': 'Pera',
      'sign_COIN': 'Barya',
      'sign_DISCOUNT': 'Diskuwento',
      'sign_EIGHT': 'Walo',
      'sign_FIVE': 'Lima',
      'sign_FOUR': 'Apat',
      'sign_HELLO': 'Kumusta',
      'sign_HOW_MANY': 'Ilan?',
      'sign_HOW_MUCH': 'Magkano?',
      'sign_NINE': 'Siyam',
      'sign_NO': 'Hindi',
      'sign_ONE': 'Isa',
      'sign_PLEASE': 'Pakiusap',
      'sign_PROBLEM': 'Problema',
      'sign_RECEIPT': 'Resibo',
      'sign_SEVEN': 'Pito',
      'sign_SIX': 'Anim',
      'sign_TEN': 'Sampu',
      'sign_THANK_YOU': 'Salamat',
      'sign_THREE': 'Tatlo',
      'sign_TWO': 'Dalawa',
      'sign_WAIT': 'Sandali',
      "sign_YOURE_WELCOME": "Walang anuman",
      'sign_WELCOME': 'Maligayang pagdating',
      'sign_YES': 'Oo',
      'sign_UNDERSTAND': 'Naiintindihan',
      'sign_DONT_UNDERSTAND': 'Hindi ko naiintindihan',
      'sign_DON’T_UNDERSTAND': 'Hindi ko naiintindihan',
      'sign_KNOW': 'Alam',
      'sign_DONT_KNOW': 'Hindi ko alam',
      'sign_DON’T_KNOW': 'Hindi ko alam',
      'sign_WRONG': 'Mali',
      'sign_CORRECT': 'Tama',
      'sign_SLOW': 'Mabagal',
      'sign_FAST': 'Mabilis',
      'sign_JANUARY': 'Enero',
      'sign_FEBRUARY': 'Pebrero',
      'sign_MARCH': 'Marso',
      'sign_APRIL': 'Abril',
      'sign_MAY': 'Mayo',
      'sign_JUNE': 'Hunyo',
      'sign_JULY': 'Hulyo',
      'sign_AUGUST': 'Agosto',
      'sign_SEPTEMBER': 'Setyembre',
      'sign_OCTOBER': 'Oktubre',
      'sign_NOVEMBER': 'Nobyembre',
      'sign_DECEMBER': 'Disyembre',
      'sign_YESTERDAY': 'Kahapon',
      'sign_FATHER': 'Tatay',
      'sign_MOTHER': 'Nanay',
      'sign_SON': 'Anak na lalaki',
      'sign_DAUGHTER': 'Anak na babae',
      'sign_GRANDFATHER': 'Lolo',
      'sign_GRANDMOTHER': 'Lola',
      'sign_UNCLE': 'Tiyo',
      'sign_AUNTIE': 'Tiya',
      'sign_COUSIN': 'Pinsan',
      'sign_PARENTS': 'Mga magulang',
      'sign_BOY': 'Batang lalaki',
      'sign_GIRL': 'Batang babae',
      'sign_MAN': 'Lalaki',
      'sign_WOMAN': 'Babae',
      'sign_DEAF': 'Bingi',
      'sign_HARD_OF_HEARING': 'Mahina ang pandinig',
      'sign_WHEELCHAIR_PERSON': 'Taong gumagamit ng wheelchair',
      'sign_WEELCHAIR_PERSON': 'Taong gumagamit ng wheelchair',
      'sign_BLIND': 'Bulag',
      'sign_DEAF_BLIND': 'Bingi at bulag',
      'sign_MARRIED': 'Kasal',
      'sign_BLUE': 'Asul',
      'sign_GREEN': 'Berde',
      'sign_RED': 'Pula',
      'sign_BROWN': 'Kayumanggi',
      'sign_BLACK': 'Itim',
      'sign_WHITE': 'Puti',
      'sign_YELLOW': 'Dilaw',
      'sign_ORANGE': 'Kahel',
      'sign_GRAY': 'Abo',
      'sign_PINK': 'Rosas',
      'sign_VIOLET': 'Lila',
      'sign_LIGHT': 'Maliwanag',
      'sign_DARK': 'Madilim',
      'sign_BREAD': 'Tinapay',
      'sign_EGG': 'Itlog',
      'sign_FISH': 'Isda',
      'sign_MEAT': 'Karne',
      'sign_CHICKEN': 'Manok',
      'sign_SPAGHETTI': 'Spaghetti',
      'sign_RICE': 'Kanin',
      'sign_LONGANISA': 'Longganisa',
      'sign_SHRIMP': 'Hipon',
      'sign_CRAB': 'Alimango',
      'sign_HOT': 'Mainit',
      'sign_COLD': 'Malamig',
      'sign_JUICE': 'Juice',
      'sign_MILK': 'Gatas',
      'sign_COFFEE': 'Kape',
      'sign_TEA': 'Tsaa',
      'sign_BEER': 'Serbesa',
      'sign_WINE': 'Alak',
      'sign_SUGAR': 'Asukal',
      'sign_NO_SUGAR': 'Walang asukal',
      'sign_A': 'A',
      'sign_B': 'B',
      'sign_C': 'C',
      'sign_D': 'D',
      'sign_E': 'E',
      'sign_F': 'F',
      'sign_G': 'G',
      'sign_H': 'H',
      'sign_I': 'I',
      'sign_J': 'J',
      'sign_K': 'K',
      'sign_L': 'L',
      'sign_M': 'M',
      'sign_N': 'N',
      'sign_O': 'O',
      'sign_P': 'P',
      'sign_Q': 'Q',
      'sign_R': 'R',
      'sign_S': 'S',
      'sign_T': 'T',
      'sign_U': 'U',
      'sign_V': 'V',
      'sign_W': 'W',
      'sign_X': 'X',
      'sign_Y': 'Y',
      'sign_Z': 'Z',
      'signRetry': 'Hindi malinaw ang senyas. Ibaba ang mga kamay at subukang muli.',
      'appName': 'Visualin',
      'tagline': 'Makita ang wika. Malayang magsalita.',
      'welcome': 'Maligayang pagdating sa Visualin',
      'skip': 'Laktawan', 'next': 'Susunod', 'getStarted': 'Magsimula',
       'onboardingSignText': 'Isalin ang FSL sa teksto',
       'onboardingSignTextBody': 'Gamitin ang iyong kamera upang gawing mababasang teksto ang Filipino Sign Language.',
       'onboardingTextSign': 'Ihanda ang teksto para sa FSL',
       'onboardingTextSignBody': 'Mag-type ng mensahe at tingnan ang salin nito sa Filipino Sign Language.',
       'onboardingSpeech': 'Pakinggan ang salin gamit ang boses',
       'onboardingSpeechBody': 'Pakinggan ang isinaling teksto nang malakas.',
       'onboardingCameraTitle': 'Kailangan ang access sa kamera',
       'onboardingCameraBody': 'Ginagamit ang access sa kamera upang makilala ang mga senyas sa FSL. Pinoproseso ang video nang live para sa pagsasalin at hindi ito sine-save. Maaari mong baguhin ang pahintulot sa kamera sa Settings anumang oras.',
       'cameraPermissionTitle': 'Kailangan ang access sa kamera',
       'cameraPermissionBody': 'Ginagamit ang access sa kamera upang makilala ang mga senyas sa FSL. Pinoproseso ang video nang live para sa pagsasalin at hindi ito sine-save. Maaari mong baguhin ang pahintulot sa kamera sa Settings anumang oras.',
      'homePrompt': 'Ano ang gusto mong isalin?',
      'fslToText': 'FSL patungong Teksto', 'fslToTextBody': 'Magsenyas gamit ang iyong kamera',
       'textToFsl': 'Teksto patungong FSL', 'textToFslBody': 'Mag-type ng teksto upang makita ang mga senyas', 'noTranslations': 'Wala pang mga salin',
       'noTranslationsBody': 'Makikita rito ang iyong mga kamakailang salin.',
      'recentTranslations': 'Mga kamakailang salin', 'home': 'Tahanan',
      'history': 'Kasaysayan',
      'flipCamera': 'Baligtarin ang kamera', 'cameraUnavailable': 'Hindi available ang kamera',
      'settings': 'Mga setting', 'outputLanguage': 'Wika ng output na teksto',
      'outputLanguageBody': 'Binabago nito ang wika ng output na teksto at boses lamang. FSL pa rin ang sign language.',
      'translation': 'Pagsasalin', 'signLanguage': 'Sign language',
      'signLanguageValue': 'Filipino Sign Language (FSL)',
      'signLanguageBody': 'FSL lamang ang kasalukuyang sinusuportahan.',
       'filipino': 'Filipino', 'english': 'Ingles', 'accessibility': 'Accessibility',
       'language': 'Wika', 'app': 'App', 'settingsSubtitle': 'I-customize ang iyong karanasan',
       'fontSize': 'Laki ng teksto', 'textSizeDescription': 'Ayusin ang laki ng teksto',
       'highContrast': 'Mataas na contrast', 'highContrastDescription': 'Palakihin ang visibility',
       'reducedMotion': 'Bawas na galaw', 'reducedMotionDescription': 'Bawasan ang mga animation',
       'cameraPreferences': 'Mga kagustuhan sa kamera', 'cameraPreferencesDescription': 'Pamahalaan ang iyong kamera',
       'helpSupport': 'Tulong at suporta', 'helpSupportDescription': 'Kumuha ng tulong sa Visualin',
       'about': 'Tungkol sa Visualin', 'aboutDescription': 'Bersyon at impormasyon ng app',
       'smallText': 'Maliit', 'defaultText': 'Default', 'largeText': 'Malaki',
       'settingsFeatureUnavailable': 'Malapit nang maging available ang option na ito.',
       'onboardingProgress': '{current} sa {total}',
      'historyStub': 'Wala pang kasaysayan ng salin.',
      'textToFslStub': 'Malapit na ang feature na teksto patungong FSL.',
      'permissionDenied': 'Hindi pinayagan ang kamera.', 'allowCamera': 'Payagan ang kamera', 'later': 'Mamaya',
      'openSettings': 'Buksan ang settings', 'handDetected': 'May nakitang kamay',
       'noHandDetected': 'Walang nakitang kamay',
       'cameraGuidanceNoHand': 'Ilagay ang iyong mga kamay sa loob ng frame.',
       'cameraGuidanceHandDetected': 'Gumawa ng isang senyas, pagkatapos ay ibaba ang mga kamay sa labas ng frame.',
       'cameraFrameLabel': 'Gabay sa posisyon ng kamay',
       'translatedText': 'Isinaling teksto',
       'translationWaiting': 'Naghihintay ng senyas…',
       'confidenceUnavailable': 'Hindi pa available ang confidence',
       'speak': 'Basahin', 'copy': 'Kopyahin', 'save': 'I-save',
       'clearText': 'Burahin', 'space': 'Espasyo',
       'fullScreen': 'Buong screen', 'exitFullScreen': 'Lumabas sa buong screen',
       'skeletonOverlayMode': 'Klasikong skeleton overlay',
       'skeletonOverlayDescription': 'Ipakita ang orihinal na mga linya ng hugpungan at buto',
       'switchToSkeletonOverlay': 'Lumipat sa skeleton overlay',
       'switchToContourOverlay': 'Lumipat sa contour overlay',
       'copiedToClipboard': 'Nakopya ang isinaling teksto',
       'fslToOutputLanguage': 'FSL patungong {language}',
       'on': 'Naka-on', 'off': 'Naka-off', 'close': 'Isara',
    },
    'en': {
      'datasetCreditBody': 'FSL-105 v2 (2023), Isaiah Jassen Tupal. This trial uses hand landmarks from selected clips.\nhttps://doi.org/10.17632/48y2y99mb9.2\nCC BY 4.0 - https://creativecommons.org/licenses/by/4.0/',
      'datasetCredit': 'FSL-105 dataset credits',
      'continuousMode': 'Continuous signing (trial)',
    'bodyMode': 'Upper-body tracking (42 / 104 signs)',
    'bodyHint': 'Keep your face, shoulders, elbows, and hands in view.',
    'bodyError': 'Upper-body tracking is unavailable. Switch it off to use hand tracking.',
    'sign_IM_FINE': "I'm fine",
    'sign_NICE_TO_MEET_YOU': "Nice to meet you",
    'sign_SEE_YOU_TOMORROW': "See you tomorrow",
    'sign_MONDAY': "Monday",
    'sign_TUESDAY': "Tuesday",
    'sign_WEDNESDAY': "Wednesday",
    'sign_THURSDAY': "Thursday",
    'sign_FRIDAY': "Friday",
    'sign_SATURDAY': "Saturday",
    'sign_SUNDAY': "Sunday",
      'continuousHint': 'Sign one word, hold its final position for a second, then start the next sign.',
      'continuousRetry': 'Unclear sign. Try that word again.',
      'continuousError': 'Could not load continuous mode. Try again.',
      'undoSign': 'Undo last sign',
      'alphabetMode': 'Alphabet Mode',
      'alphabetHint': 'Hold a steady FSL handshape or trace J/Z.',
      'alphabetRetry': 'Unclear letter. Hold your handshape steady and try again.',
      'alphabetError': 'Could not load alphabet model. Try again.',
      'sign_GOOD_MORNING': 'Good morning',
      'sign_GOOD_AFTERNOON': 'Good afternoon',
      'sign_HOW_ARE_YOU': 'How are you?',
      'sign_TODAY': 'Today',
      'sign_TOMORROW': 'Tomorrow',
      'sign_AGAIN': 'Again',
      'sign_CARD': 'Card',
      'sign_CASH': 'Cash',
      'sign_COIN': 'Coin',
      'sign_DISCOUNT': 'Discount',
      'sign_EIGHT': 'Eight',
      'sign_FIVE': 'Five',
      'sign_FOUR': 'Four',
      'sign_HELLO': 'Hello',
      'sign_HOW_MANY': 'How many?',
      'sign_HOW_MUCH': 'How much?',
      'sign_NINE': 'Nine',
      'sign_NO': 'No',
      'sign_ONE': 'One',
      'sign_PLEASE': 'Please',
      'sign_PROBLEM': 'Problem',
      'sign_RECEIPT': 'Receipt',
      'sign_SEVEN': 'Seven',
      'sign_SIX': 'Six',
      'sign_TEN': 'Ten',
      'sign_THANK_YOU': 'Thank you',
      'sign_THREE': 'Three',
      'sign_TWO': 'Two',
      'sign_WAIT': 'Wait',
      "sign_YOURE_WELCOME": "You're welcome",
      'sign_WELCOME': 'Welcome',
      'sign_YES': 'Yes',
      'sign_UNDERSTAND': 'Understand',
      'sign_DONT_UNDERSTAND': "Don't understand",
      'sign_DON’T_UNDERSTAND': "Don't understand",
      'sign_KNOW': 'Know',
      'sign_DONT_KNOW': "Don't know",
      'sign_DON’T_KNOW': "Don't know",
      'sign_WRONG': 'Wrong',
      'sign_CORRECT': 'Correct',
      'sign_SLOW': 'Slow',
      'sign_FAST': 'Fast',
      'sign_JANUARY': 'January',
      'sign_FEBRUARY': 'February',
      'sign_MARCH': 'March',
      'sign_APRIL': 'April',
      'sign_MAY': 'May',
      'sign_JUNE': 'June',
      'sign_JULY': 'July',
      'sign_AUGUST': 'August',
      'sign_SEPTEMBER': 'September',
      'sign_OCTOBER': 'October',
      'sign_NOVEMBER': 'November',
      'sign_DECEMBER': 'December',
      'sign_YESTERDAY': 'Yesterday',
      'sign_FATHER': 'Father',
      'sign_MOTHER': 'Mother',
      'sign_SON': 'Son',
      'sign_DAUGHTER': 'Daughter',
      'sign_GRANDFATHER': 'Grandfather',
      'sign_GRANDMOTHER': 'Grandmother',
      'sign_UNCLE': 'Uncle',
      'sign_AUNTIE': 'Auntie',
      'sign_COUSIN': 'Cousin',
      'sign_PARENTS': 'Parents',
      'sign_BOY': 'Boy',
      'sign_GIRL': 'Girl',
      'sign_MAN': 'Man',
      'sign_WOMAN': 'Woman',
      'sign_DEAF': 'Deaf',
      'sign_HARD_OF_HEARING': 'Hard of hearing',
      'sign_WHEELCHAIR_PERSON': 'Wheelchair person',
      'sign_WEELCHAIR_PERSON': 'Wheelchair person',
      'sign_BLIND': 'Blind',
      'sign_DEAF_BLIND': 'Deaf-blind',
      'sign_MARRIED': 'Married',
      'sign_BLUE': 'Blue',
      'sign_GREEN': 'Green',
      'sign_RED': 'Red',
      'sign_BROWN': 'Brown',
      'sign_BLACK': 'Black',
      'sign_WHITE': 'White',
      'sign_YELLOW': 'Yellow',
      'sign_ORANGE': 'Orange',
      'sign_GRAY': 'Gray',
      'sign_PINK': 'Pink',
      'sign_VIOLET': 'Violet',
      'sign_LIGHT': 'Light',
      'sign_DARK': 'Dark',
      'sign_BREAD': 'Bread',
      'sign_EGG': 'Egg',
      'sign_FISH': 'Fish',
      'sign_MEAT': 'Meat',
      'sign_CHICKEN': 'Chicken',
      'sign_SPAGHETTI': 'Spaghetti',
      'sign_RICE': 'Rice',
      'sign_LONGANISA': 'Longanisa',
      'sign_SHRIMP': 'Shrimp',
      'sign_CRAB': 'Crab',
      'sign_HOT': 'Hot',
      'sign_COLD': 'Cold',
      'sign_JUICE': 'Juice',
      'sign_MILK': 'Milk',
      'sign_COFFEE': 'Coffee',
      'sign_TEA': 'Tea',
      'sign_BEER': 'Beer',
      'sign_WINE': 'Wine',
      'sign_SUGAR': 'Sugar',
      'sign_NO_SUGAR': 'No sugar',
      'sign_A': 'A',
      'sign_B': 'B',
      'sign_C': 'C',
      'sign_D': 'D',
      'sign_E': 'E',
      'sign_F': 'F',
      'sign_G': 'G',
      'sign_H': 'H',
      'sign_I': 'I',
      'sign_J': 'J',
      'sign_K': 'K',
      'sign_L': 'L',
      'sign_M': 'M',
      'sign_N': 'N',
      'sign_O': 'O',
      'sign_P': 'P',
      'sign_Q': 'Q',
      'sign_R': 'R',
      'sign_S': 'S',
      'sign_T': 'T',
      'sign_U': 'U',
      'sign_V': 'V',
      'sign_W': 'W',
      'sign_X': 'X',
      'sign_Y': 'Y',
      'sign_Z': 'Z',
      'signRetry': 'Unclear sign. Lower your hands and try again.',
      'appName': 'Visualin',
      'tagline': 'See language. Speak freely.', 'welcome': 'Welcome to Visualin',
      'skip': 'Skip', 'next': 'Next', 'getStarted': 'Get started',
       'onboardingSignText': 'Translate FSL into text',
       'onboardingSignTextBody': 'Use your camera to turn Filipino Sign Language into readable text.',
       'onboardingTextSign': 'Prepare text for FSL',
       'onboardingTextSignBody': 'Type a message and view its Filipino Sign Language translation.',
       'onboardingSpeech': 'Listen to translations with speech',
       'onboardingSpeechBody': 'Hear translated text spoken aloud.',
       'onboardingCameraTitle': 'Camera access is required',
       'onboardingCameraBody': 'Camera access is used to recognize FSL signs. Video is processed live for translation and is not saved. You can change camera permission later in Settings.',
       'cameraPermissionTitle': 'Camera access is required',
       'cameraPermissionBody': 'Camera access is used to recognize FSL signs. Video is processed live for translation and is not saved. You can change camera permission later in Settings.',
      'homePrompt': 'What would you like to translate?',
      'fslToText': 'FSL to Text', 'fslToTextBody': 'Sign using your camera',
       'textToFsl': 'Text to FSL', 'textToFslBody': 'Type text to view signs', 'noTranslations': 'No translations yet',
       'noTranslationsBody': 'Your recent translations will appear here.',
      'recentTranslations': 'Recent translations', 'home': 'Home',
      'history': 'History', 'flipCamera': 'Flip camera',
      'cameraUnavailable': 'Camera unavailable', 'settings': 'Settings',
      'outputLanguage': 'Output text language',
      'outputLanguageBody': 'This changes output text and speech only. The sign language remains FSL.',
      'translation': 'Translation', 'signLanguage': 'Sign language',
      'signLanguageValue': 'Filipino Sign Language (FSL)',
      'signLanguageBody': 'Currently, only FSL is supported.',
       'filipino': 'Filipino', 'english': 'English', 'accessibility': 'Accessibility',
       'language': 'Language', 'app': 'App', 'settingsSubtitle': 'Customize your experience',
       'fontSize': 'Text size', 'textSizeDescription': 'Adjust text size',
       'highContrast': 'High contrast', 'highContrastDescription': 'Increase visibility',
       'reducedMotion': 'Reduced motion', 'reducedMotionDescription': 'Minimize animations',
       'cameraPreferences': 'Camera preferences', 'cameraPreferencesDescription': 'Manage your camera',
       'helpSupport': 'Help and support', 'helpSupportDescription': 'Get help with Visualin',
       'about': 'About Visualin', 'aboutDescription': 'App version and information',
       'smallText': 'Small', 'defaultText': 'Default', 'largeText': 'Large',
       'settingsFeatureUnavailable': 'This option will be available soon.',
       'onboardingProgress': '{current} of {total}',
      'historyStub': 'No translation history yet.', 'textToFslStub': 'Text to FSL is coming soon.',
      'permissionDenied': 'Camera permission was not granted.', 'allowCamera': 'Allow camera', 'later': 'Later',
      'openSettings': 'Open settings', 'handDetected': 'Hand detected',
       'noHandDetected': 'No hand detected',
       'cameraGuidanceNoHand': 'Move your hands inside the frame.',
       'cameraGuidanceHandDetected': 'Make one sign, then lower your hands out of view.',
       'cameraFrameLabel': 'Hand position guide',
       'translatedText': 'Translated text',
       'translationWaiting': 'Waiting for a sign…',
       'confidenceUnavailable': 'Confidence unavailable',
       'speak': 'Speak', 'copy': 'Copy', 'save': 'Save',
       'clearText': 'Clear', 'space': 'Space',
       'fullScreen': 'Full screen', 'exitFullScreen': 'Exit full screen',
       'skeletonOverlayMode': 'Classic skeleton overlay',
       'skeletonOverlayDescription': 'Show the original joint and bone skeleton lines',
       'switchToSkeletonOverlay': 'Switch to skeleton overlay',
       'switchToContourOverlay': 'Switch to contour overlay',
       'copiedToClipboard': 'Copied translated text',
       'fslToOutputLanguage': 'FSL to {language}',
       'on': 'On', 'off': 'Off', 'close': 'Close',
    },
  };
}

class _AppLocalizationsDelegate extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();
  @override bool isSupported(Locale locale) => ['fil', 'en'].contains(locale.languageCode);
  @override Future<AppLocalizations> load(Locale locale) async => AppLocalizations(locale);
  @override bool shouldReload(_AppLocalizationsDelegate old) => false;
}
