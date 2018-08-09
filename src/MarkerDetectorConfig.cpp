/*******************************************************************************
**
** Copyright (C) 2015-2019 EPFL (Swiss Federal Institute of Technology)
**
** Contact:
**   Dr. Davide A. Cucci, post-doctoral researcher
**   E-mail: davide.cucci@epfl.ch
**
**   Geodetic Engineering Laboratory,
**   1015 Lausanne, Switzerland (www.topo.epfl.ch).
**
**
**
**   This file is part of visiona.
**
**   visiona is free software: you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation, either version 3 of the License, or
**   (at your option) any later version.
**
**   visiona is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with visiona.  If not, see <http://www.gnu.org/licenses/>.
**
*******************************************************************************/

/*
 * MarkerDetectorConfig.cpp
 *
 *  Created on: Mar 20, 2015
 *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
 *
 *  Modified on: Aug 30, 2018
 *      By : Maxime Werlen
 */

#include "MarkerDetectorConfig.h"

#include <libconfig.h++>

using namespace std;
using namespace libconfig;

namespace visiona {

MarkerDetectorConfig::MarkerDetectorConfig() :
  K(3, 3, CV_64FC1, cv::Scalar(0)), distortion(5, 1, CV_64FC1, cv::Scalar(0)) {

  CannyBlurKernelSize = 3;
  CannyLowerThreshold = 15;
  CannyHigherThreshold = 75;

  contourFilterMinSize = 17;

  markerxCorrThreshold = 0.60;

}

bool MarkerDetectorConfig::loadConfig(const std::string &file) {

  Config cf;

  try {
    cf.readFile(file.c_str());
  } catch (const FileIOException &fioex) {
    cerr << " * ERROR: I/O error while reading " << file << endl;
    return false;
  } catch (const ParseException &pex) {
    cerr << " * ERROR: malformed cfg file at " << pex.getFile() << ":"
        << pex.getLine() << " - " << pex.getError() << endl;
    return false;
  }

  const Setting &root = cf.getRoot();

  try {
    Setting &camera = root["camera"];

    double f;

    camera.lookupValue("f", f);

    K.at<double>(0, 0) = f;
    K.at<double>(0, 2) = camera["pp"][0];
    K.at<double>(1, 1) = f;
    K.at<double>(1, 2) = camera["pp"][1];
    K.at<double>(2, 2) = 1.0;

    distortion.at<double>(0, 0) = camera["RD"][0];
    distortion.at<double>(1, 0) = camera["RD"][1];
    distortion.at<double>(2, 0) = camera["TD"][0];
    distortion.at<double>(3, 0) = camera["TD"][1];
    distortion.at<double>(4, 0) = camera["RD"][2];

    Setting &targets = root["targets"];

    targets.lookupValue("d", markerDiameter);
    targets.lookupValue("id", markerInnerDiameter);

    fromSettingToMarkers(targets["targetModels"], markerModels);
    targets.lookupValue("signalRadiusPercentage", markerSignalRadiusPercentage);

  } catch (SettingNotFoundException &e) {
    cerr << " * ERROR: \"" << e.getPath() << "\" not defined." << endl;
    return false;
  }

  return true;

}

}
