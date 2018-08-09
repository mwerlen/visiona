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
 * batchdetector.cpp
 *
 *  Created on: May 13, 2015
 *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
 *
 *  Edited on: Aug. 2018
 *      By Maxime Werlen
 */

#include <fstream>
#include <iostream>
#include <vector>

#include <dirent.h>
#include <unistd.h>
#include <getopt.h>
#include "opencv2/highgui/highgui.hpp"
#include <libconfig.h++>

#include "Visiona.h"

using namespace std;
using namespace cv;
using namespace libconfig;
using namespace visiona;

#define OPTPATHSET 1
#define OPTEXTSET 2
#define OPTCFGFILESET 4
#define OPTDEBUG 8
#define OPTPREFIXSET 16
#define OPTUSESEEDPOINTS 32 

int main(int argc, char *argv[]) {

  // --------------------- parse command line arguments ------------------------

  static struct option long_options[] = {
      { "path", required_argument, 0, 'p' },
      { "ext", required_argument, 0, 'e' },
      { "config", required_argument, 0, 'c' },
      { "debug", no_argument, 0, 'd' },
      { "prefix", required_argument, 0, 'f' },
      { 0, 0, 0, 0 }
  };

  unsigned int optionflag = 0;
  char *imagePath, *imgext, *configpath, *prefix, *detectioncfgpath, *seedpointspath;

  opterr = 0;
  int c;
  while ((c = getopt_long_only(argc, argv, "", long_options, NULL)) != -1) {
    switch (c) {
    case 'p':
      if (optarg[strlen(optarg) - 1] == '/') {
        optarg[strlen(optarg) - 1] = 0;
      }
      imagePath = optarg;
      optionflag |= OPTPATHSET;
      break;
    case 'e':
      imgext = optarg;
      optionflag |= OPTEXTSET;
      break;
    case 'c':
      configpath = optarg;
      optionflag |= OPTCFGFILESET;
      break;
    case 'd':
      optionflag |= OPTDEBUG;
      break;
    case 'f':
      prefix = optarg;
      optionflag |= OPTPREFIXSET;
      break;
    case '?':
      cerr << " * ERROR: unknown option or missing argument" << endl;
      return 1;

    default:
      abort();
    }
  }

  if ((optionflag & OPTPATHSET) == 0) {
    cerr << " * ERROR: image path not specified (-p)" << endl;
    return 1;
  }
  if ((optionflag & OPTEXTSET) == 0) {
    cerr << " * ERROR: image extension not specified (-e)" << endl;
    return 1;
  }
  if ((optionflag & OPTCFGFILESET) == 0) {
    cerr << " * ERROR: config file not specified (-c)" << endl;
    return 1;
  }

  DIR *dir;
  if ((dir = opendir(imagePath)) == NULL) {
    cerr << " * ERROR: could not open image folder" << endl;
    return 1;
  }

  // --------------------- configuration ---------------------------------------

  // loading config
  MarkerDetectorConfig cfg;
  if (!cfg.loadConfig(configpath)) {
    return 1;
  }

  MarkerDetector *detector = MarkerDetectorFactory::makeMarkerDetector(cfg);

  // --------------------- generating image list -------------------------------

  vector<string> images;

  struct dirent *file;

  while ((file = readdir(dir)) != NULL) {
    if (strcmp(file->d_name + strlen(file->d_name) - 3, imgext) == 0) {

      string fn_prefix = "_";
      if (optionflag & OPTPREFIXSET) {
        fn_prefix = string(prefix);
      }

      string fname(file->d_name);

      int sep = fname.find(fn_prefix);
      if (sep != string::npos) {
        images.push_back(file->d_name);
      }
    }
  }

  closedir(dir);

  // --------------------- prepare output files --------------------------------
  ofstream *output = new ofstream("output.log");

  cout << images.size() << " file(s) found." << endl;

  // --------------------- process every image ---------------------------------
  for (int i = 0; i < images.size(); i++) {
    string filename = images[i];

    cout << filename << " ..." << endl;

    string imgName = imagePath + string("/") + filename;

    Mat raw = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);

    // --- real detection proces starts here

    vector<Target> returnedValues = detector->detect(raw);
    for (auto targetIt = returnedValues.begin(); targetIt != returnedValues.end(); ++targetIt) {
      Target &target = *targetIt;

      if (target.detected) {
        detector->evaluateExposure(raw, &target);
        detector->measureRough(raw, &target);
        detector->measure(raw, &target);
      }

      // --- and ends here

      // --- where output is produced

      // TODO: introduce a flag to enable/disable this

      if (target.roughlyMeasured) {

        for (int j= 0; j < target.markerModel->signalModel.size() / 2; ++j) {
          double x = target.codePoints[j].x, y = target.codePoints[j].y;

          // convert to photogrammetry convention
          // TODO: put an option
          if (true) {
            swap(x, y);
            x = -(x - raw.rows / 2.0) * 4.7e-3; // MWE : C'est quoi ce truc ?
            y = -(y - raw.cols / 2.0) * 4.7e-3;
          }

          *output << filename.substr(0, filename.length() - 4) << " -> x:";
          *output << fixed << setprecision(6) << x << " - y:";
          *output << fixed << setprecision(6) << y << endl;
          output->flush();
        }
      }
    }
  }

  return 0;
}

