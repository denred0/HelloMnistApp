import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:hello_mnist/utils/constants.dart';
import 'package:hello_mnist/dl_model/classifier.dart';
import 'dart:io';

class UploadImage extends StatefulWidget {
  @override
  _UploadImageState createState() => _UploadImageState();
}

class _UploadImageState extends State<UploadImage> {
  final picker = ImagePicker();
  Classifier classifier = Classifier();
  late PickedFile image;
  int digit = -1;

  @override
  Widget build(BuildContext context) {
    SizeConfig().init(context);
    // SizeConfig._mediaQueryData;

    return Scaffold(
      backgroundColor: backgroundColor,
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.deepOrangeAccent,
        child: Icon(Icons.camera_alt_outlined),
        onPressed: () async {
          image = await picker.getImage(
            source: ImageSource.gallery,
            maxHeight: 300,
            maxWidth: 300,
            imageQuality: 100,
          );
          // digit = 1;
          digit = await classifier.classifyImage(image);
          setState(() {});
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
            Text("Image will be shown below",
                style: TextStyle(fontSize: 20, color: textColor)),
            SizedBox(
              height: SizeConfig.safeBlockVertical * 2,
            ),
            Container(
              width: SizeConfig.safeBlockVertical * 40 + borderSize * 2,
              height: SizeConfig.safeBlockVertical * 40 + borderSize * 2,
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(color: Colors.grey, width: borderSize),
                image: DecorationImage(
                  fit: BoxFit.fill,
                  image: digit == -1
                      ? AssetImage('assets/white_background.jpg')
                          as ImageProvider
                      : FileImage(File(image.path)),
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
