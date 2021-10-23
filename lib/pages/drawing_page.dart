import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:hello_mnist/dl_model/classifier.dart';
import 'package:hello_mnist/utils/constants.dart';

class DrawPage extends StatefulWidget {
  @override
  _DrawPageState createState() => _DrawPageState();
}

class _DrawPageState extends State<DrawPage> {
  Classifier _classifier = Classifier();
  List<Offset> pointsDrawing = List<Offset>.empty(growable: true);
  List<Offset> pointsPredict = List<Offset>.empty(growable: true);
  final pointMode = ui.PointMode.points;
  int digit = -1;

  @override
  Widget build(BuildContext context) {
    SizeConfig().init(context);

    return Scaffold(
      backgroundColor: backgroundColor,
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.deepOrangeAccent,
        child: Icon(Icons.close),
        onPressed: () {
          setState(() {
            pointsDrawing.clear();
            pointsPredict.clear();
            digit = -1;
          });
        },
      ),
      appBar: AppBar(
        backgroundColor: Colors.blue,
        title: Text("Digit recognizer"),
      ),
      body: Align(
        alignment: Alignment.center,
        child: Column(
          children: [
            SizedBox(
              height: SizeConfig.safeBlockVertical * 5,
            ),
            Text("Draw digit inside the box", style: TextStyle(fontSize: 20)),
            SizedBox(
              height: SizeConfig.safeBlockVertical * 2,
            ),
            Container(
              width: canvasSize,
              height: canvasSize,
              decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.grey, width: borderSize)),
              child: GestureDetector(
                onPanUpdate: (DragUpdateDetails details) {
                  Offset _localPosition = details.localPosition;
                  if (_localPosition.dx >= 0 &&
                      _localPosition.dx <= canvasSize &&
                      _localPosition.dy >= 0 &&
                      _localPosition.dy <= canvasSize) {
                    setState(() {
                      // print(_localPosition.dx);
                      pointsDrawing.add(_localPosition);

                      pointsPredict.add(_localPosition);
                      // if (_localPosition.dx - 1 > 0 &&
                      //     _localPosition.dx + 1 < canvasSize) {
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx - 1, _localPosition.dy));
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx + 1, _localPosition.dy));
                      // } else if (_localPosition.dx + 1 > canvasSize) {
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx - 1, _localPosition.dy));
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx - 2, _localPosition.dy));
                      // } else if (_localPosition.dx - 1 < 0) {
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx + 1, _localPosition.dy));
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx + 2, _localPosition.dy));
                      // }

                      // if (_localPosition.dy - 1 > 0 &&
                      //     _localPosition.dy + 1 < canvasSize) {
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx, _localPosition.dy - 1));
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx, _localPosition.dy + 1));
                      // } else if (_localPosition.dy + 1 > canvasSize) {
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx, _localPosition.dy - 1));
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx, _localPosition.dy - 2));
                      // } else if (_localPosition.dy - 1 < 0) {
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx, _localPosition.dy + 1));
                      //   pointsDrawing.add(
                      //       Offset(_localPosition.dx, _localPosition.dy + 2));
                      // }
                    });
                  }
                },
                onPanEnd: (DragEndDetails details) async {
                  print('press end');
                  pointsDrawing.add(Offset(-1, -1));
                  digit = await _classifier.classifyDrawing(pointsPredict);
                  setState(() {});
                },
                child: CustomPaint(
                  painter: Painter(points: pointsDrawing),
                ),
              ),
            ),
            SizedBox(
              height: SizeConfig.safeBlockVertical * 7,
            ),
            Text("Current Prediction:",
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: textColor)),
            SizedBox(
              height: SizeConfig.safeBlockVertical * 3,
            ),
            Text(digit == -1 ? "" : "$digit",
                style: TextStyle(
                    fontSize: 40,
                    fontWeight: FontWeight.bold,
                    color: textColor)),
          ],
        ),
      ),
    );
  }
}

class Painter extends CustomPainter {
  final List<Offset> points;

  Painter({required this.points});

  final Paint _paintDetails = Paint()
    ..style = PaintingStyle.stroke
    ..strokeWidth =
        4.0 // strokeWidth 4 looks good, but strokeWidth approx. 16 looks closer to training data
    ..color = Colors.black;

  @override
  void paint(Canvas canvas, Size size) {
    for (int i = 0; i < points.length - 1; i++) {
      if (points[i + 1] != Offset(-1.0, -1.0) &&
          points[i] != Offset(-1.0, -1.0)) {
        canvas.drawLine(points[i], points[i + 1], _paintDetails);
      }
    }
  }

  @override
  bool shouldRepaint(Painter oldDelegate) {
    return true;
  }
}
