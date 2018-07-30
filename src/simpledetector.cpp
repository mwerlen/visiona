/*
 * simpledetector.cpp
 *
 *  Created on: August 30, 2018
 *      Author: Maxime Werlen
 * 
 *  Based on work from Davide A. Cucci (davide.cucci@epfl.ch)
 */

#include <fstream>
#include <iostream>
#include <vector>

#include <unistd.h>

#include "opencv2/highgui/highgui.hpp"

#include <libconfig.h++>

#include "Visiona.h"

using namespace std;
using namespace cv;
using namespace libconfig;
using namespace visiona;

int main(int argc, char *argv[]) {

  // config
  MarkerDetectorConfig config;
  config.loadConfig("../config/simple.cfg");

  // Initialize detector
  MarkerDetector *markerDetector = MarkerDetectorFactory::makeMarkerDetector(config);
  
  // Image
  Mat raw = imread("../target/test.png", CV_LOAD_IMAGE_GRAYSCALE);

  // --- real detection proces starts here
  shared_ptr<Target> target;

  vector<shared_ptr<Target>> returnedValue = markerDetector->detect(raw);
  target = returnedValue[0];

  if (target->detected) {
    markerDetector->evaluateExposure(raw, target);
    markerDetector->measureRough(raw, target);
    cout << "Target (roughly):" << endl;
    cout << "center x:" << fixed << setprecision(6) << target->outer.center.x << " ";
    cout << "- y: "  << fixed << setprecision(6) << target->outer.center.y << endl;

    markerDetector->measure(raw, target);
    cout << "Target:" << endl;
    cout << "center x:" << fixed << setprecision(6) << target->outer.center.x << " ";
    cout << "- y: "  << fixed << setprecision(6) << target->outer.center.y << endl;

  }

  // --- where output is produced


  if (target->roughlyMeasured) {

    cout << "Code points:" << endl;

    for (int i = 0; i < config.markerSignalModel.size() / 2; ++i) {
      double x = target->codePoints[i].x, y = target->codePoints[i].y;

      // convert to photogrammetry convention
      // TODO: put an option
      if (true) {
        swap(x, y);
        x = -(x - raw.rows / 2.0) * 4.7e-3;
        y = -(y - raw.cols / 2.0) * 4.7e-3;
      }

      cout << "-> x:" << fixed << setprecision(6) << x << " ";
      cout << " - y: " << fixed << setprecision(6) << y << endl;
    }
  }

  return 0;
}
