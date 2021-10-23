import 'package:flutter/widgets.dart';
import 'package:hello_mnist/utils/constants.dart';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:ui' as ui;
import 'dart:io' as io;
import 'package:image/image.dart' as img;
import 'package:path/path.dart';

class Classifier {
  Classifier(); // Empty init/constructor

  classifyImage(PickedFile image) async {
    // Takes PickedFile image as input and returns an integer
    // of which integer it was (hopefully)!

    // Ugly boilerplate to get it to Uint8List
    var _file = io.File(image.path);
    img.Image imageTemp = img.decodeImage(_file.readAsBytesSync());
    img.Image resizedImg =
        img.copyResize(imageTemp, height: mnistSize, width: mnistSize);
    var imgBytes = resizedImg.getBytes();
    var imgAsList = imgBytes.buffer.asUint8List();

    // Everything "important" is done in getPred
    return getPred(imgAsList);
  }

  classifyDrawing(List<Offset> points) async {
    // Takes img as a List of Points from Drawing and returns Integer
    // of which digit it was (hopefully)!

    // Ugly boilerplate to get it to Uint8List
    final picture = toPicture(points);
    // convert List to Picture
    final image =
        await picture.toImage(mnistSize, mnistSize); // Picture to 28x28 Image

    ByteData? imgBytes = await image.toByteData(); // Read this image
    // final pngBytes = await image.toByteData(format: new ui.EncodingFormat.png());
    // new Image.memory(new Uint8List.view(imgBytes!.buffer));

    var imgAsList = imgBytes!.buffer.asUint8List();

    // Everything "important" is done in getPred
    return getPred(imgAsList);
  }

  Future<int> getPred(Uint8List imgAsList) async {
    // Takes img as a List as input and returns an Integer of which digit
    // the model predicts.

    // We need to convert Image which is in RGBA to Grayscale, first we can ignore
    // the alpha (opacity) and we can take the mean of R,G,B into a single channel
    final resultBytes = List<double>.filled(mnistSize * mnistSize, 0);

    int index = 0;
    for (int i = 0; i < imgAsList.lengthInBytes; i += 4) {
      final r = imgAsList[i];
      final g = imgAsList[i + 1];
      final b = imgAsList[i + 2];

      // Take the mean of R,G,B channel into single GrayScale
      resultBytes[index] = (((r + g + b) / 3.0) / 255.0);
      index++;
    }

    // Thanks for having inbuilt reshape in tflite flutter, this would much more
    // annoying otherwise :)
    var input = resultBytes.reshape([1, 28, 28, 1]);
    var output = List.filled(1 * 10, 0).reshape([1, 10]);

    printWrapped(input.toString());

    // for (int i = 0; i < 28; i++) {
    //   for (int j = 0; j < 28; j++) {
    //     if (input[0][i][j][0] > 0) {
    //       input[0][i][j][0] = 1.0;
    //     }
    //   }
    // }

    printWrapped(input.toString());

    // Can be used to set GPUDelegate, NNAPI, parallel cores etc. We won't use
    // this, but can be good to know it exists.
    InterpreterOptions interpreterOptions = InterpreterOptions();

    // Track how long it took to do inference
    int startTime = new DateTime.now().millisecondsSinceEpoch;

    try {
      Interpreter interpreter = await Interpreter.fromAsset("model.tflite",
          options: interpreterOptions);
      interpreter.run(input, output);
    } catch (e) {
      print('Error loading or running model: ' + e.toString());
    }



    int endTime = new DateTime.now().millisecondsSinceEpoch;
    // print("Inference took ${endTime - startTime} ms");

    // Obtain the highest score from the output of the model
    double highestProb = 0;
    int digitPred = -1;

    for (int i = 0; i < output[0].length; i++) {
      if (output[0][i] > highestProb) {
        highestProb = output[0][i];
        digitPred = i;
      }
    }
    return digitPred;
  }
}

void printWrapped(String text) {
  final pattern = RegExp('.{1,800}'); // 800 is the size of each chunk
  pattern.allMatches(text).forEach((match) => print(match.group(0)));
}

ui.Picture toPicture(List<Offset> points) {
  // Obtain a Picture from a List of points
  // This Picture can then be converted to something
  // we can send to our model. Seems unnecessary to draw twice,
  // but couldn't find a way to record while using CustomPainter,
  // this is a future improvement to make.

  final _whitePaint = Paint()
    ..strokeCap = StrokeCap.round
    ..color = Colors.white
    ..strokeWidth = strokeWidth;

  final _bgPaint = Paint()..color = Colors.black;
  final _canvasCullRect = Rect.fromPoints(
      Offset(0, 0), Offset(mnistSize.toDouble(), mnistSize.toDouble()));
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder, _canvasCullRect)
    ..scale(mnistSize / canvasSize);

  canvas.drawRect(Rect.fromLTWH(0, 0, 28, 28), _bgPaint);

  for (int i = 0; i < points.length - 1; i++) {
    // if (points[i] != null && points[i + 1] != null) {
    canvas.drawLine(points[i], points[i + 1], _whitePaint);
    // }
  }

  return recorder.endRecording();
}
