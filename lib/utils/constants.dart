import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';

// For main page BottomNavigator
final double iconSize = 30;
final double selectedFontSize = 14;
final double unselectedFontSize = 14;

// For Drawing Canvas
final double canvasSize = SizeConfig.safeBlockVertical * 40 + borderSize * 2;
final double borderSize = 2;
final double strokeWidth = 16;
final int mnistSize = 28;

final backgroundColor = Colors.grey[200];
final textColor = Colors.grey[800];

class SizeConfig {
  static late MediaQueryData _mediaQueryData;
  static double screenWidth = 0;
  static double screenHeight = 0;
  static double blockSizeHorizontal = 0;
  static double blockSizeVertical = 0;
  static double _safeAreaHorizontal = 0;
  static double _safeAreaVertical = 0;
  static double safeBlockHorizontal = 0;
  static double safeBlockVertical = 0;

  void init(BuildContext context) {
    _mediaQueryData = MediaQuery.of(context);
    screenWidth = _mediaQueryData.size.width;
    screenHeight = _mediaQueryData.size.height;
    blockSizeHorizontal = screenWidth / 100;
    blockSizeVertical = screenHeight / 100;
    _safeAreaHorizontal =
        _mediaQueryData.padding.left + _mediaQueryData.padding.right;
    _safeAreaVertical =
        _mediaQueryData.padding.top + _mediaQueryData.padding.bottom;
    safeBlockHorizontal = (screenWidth - _safeAreaHorizontal) / 100;
    safeBlockVertical = (screenHeight - _safeAreaVertical) / 100;
  }
}
