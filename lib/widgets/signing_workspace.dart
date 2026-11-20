import 'package:flutter/material.dart';

/// Gives the live camera a stable viewport as captions and controls grow.
class SigningWorkspace extends StatelessWidget {
  const SigningWorkspace(
      {super.key, required this.preview, required this.controls});
  final Widget preview;
  final List<Widget> controls;

  @override
  Widget build(BuildContext context) {
    final details = ListView(padding: EdgeInsets.zero, children: controls);
    return LayoutBuilder(builder: (context, limits) {
      if (limits.maxWidth >= 540 && limits.maxWidth > limits.maxHeight) {
        return Row(children: [
          Expanded(flex: 3, child: SizedBox.expand(child: preview)),
          const SizedBox(width: 12),
          Expanded(flex: 2, child: details),
        ]);
      }
      return Column(children: [
        Expanded(flex: 6, child: SizedBox.expand(child: preview)),
        const SizedBox(height: 8),
        Expanded(flex: 4, child: details),
      ]);
    });
  }
}
