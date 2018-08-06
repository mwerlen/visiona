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
 * MarkerDetector.cpp
 *
 *  Created on: Mar 20, 2015
 *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
 *
 *  Modified on: Aug 30, 2018
 *      By: Maxime Werlen
 */

#include "MarkerDetector_impl.h"

#include <thread>

#include <Eigen/Eigenvalues>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

namespace visiona {

/*
 * Constructor initialise dot positions (_worldPoints) from config
 *
 */
MarkerDetector_impl::MarkerDetector_impl(const MarkerDetectorConfig &cfg) : MarkerDetector(cfg) {

  for (int cnt = 0; cnt < _cfg.markerSignalModel.size() / 2; ++cnt) {
    int i = (_cfg.markerSignalStartsWith == 1.0 ? 0 : 1) + 2 * cnt;

    float maxAngle, minAngle, angle;

    if (i == 0) {
      minAngle = 2 * M_PI
          * (_cfg.markerSignalModel[_cfg.markerSignalModel.size() - 1] - 1);
    } else {
      minAngle = 2 * M_PI * _cfg.markerSignalModel[i - 1];
    }

    maxAngle = 2 * M_PI * _cfg.markerSignalModel[i];

    angle = 0.5 * (maxAngle + minAngle);

    Point3f wp;
    wp.x = cos(angle);
    wp.y = sin(angle);
    wp.z = 0.0;
    wp *= _cfg.markerDiameter * _cfg.markerSignalRadiusPercentage / 2.0;

    _worldPoints.push_back(wp);
  }

}


/*
 * detect method do the following steps :
 *   - detecting edges
 *   - starting with edges detecting contours
 *   - starting with contours detecting circles
 *   - starting with circles detecting clusters (set of two concentric circles)
 *   - Best matching circle (with respect to the reference signal) is selected and theta angle computed
 * Heading (angle between ref and signal), Inner and Outer circle are set.
 */
bool MarkerDetector_impl::detect(const cv::Mat &raw, Circle &outer, Circle &inner, float &heading) {

  Mat edges;
  detectEdges(raw, edges);

  Contours ctr;
  detectContours(edges, ctr);

  Circles circles;
  filterContours(ctr, circles);

  std::vector<CircleCluster> clusters;
  clusterCircles(circles, clusters);

  bool found = false;

  if (circles.size() > 0) {
    // Select marker among the cluster representatives
    std::vector<int> representatives;
    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
      representatives.push_back(it->rep);
    }

    int selectedCluster;
    found = selectMarker(raw, circles, representatives, selectedCluster,
        heading);

    if (found) {
      // in which we have exactly two circles per cluster
      assert(clusters[selectedCluster].circleIds.size() == 2);

      outer = circles[clusters[selectedCluster].circleIds[0]];
      inner = circles[clusters[selectedCluster].circleIds[1]];
    }
  }

  return found;
}

/*
 * Measure method will compute ellipse instead of circles and do computations again
 * to get target center and distance
 *
 */
bool MarkerDetector_impl::measure(const cv::Mat &image, std::shared_ptr<Target> tg) {

  // fit ellipses on the markers
  Ellipse outerElps, innerElps;
  fitEllipse(tg->outer.cnt, outerElps);
  fitEllipse(tg->inner.cnt, innerElps);

  // refine with subpixel
  vector<Point2f> outerElpsCntSubpx;
  vector<double> outerAngles;
  refineEllipseCntWithSubpixelEdges(image, *tg, outerElps, true, 2, outerElpsCntSubpx, outerAngles);

  vector<Point2f> innerElpsCntSubpx;
  vector<double> innerAngles;
  refineEllipseCntWithSubpixelEdges(image, *tg, innerElps, true, 2, innerElpsCntSubpx, innerAngles);

  // undistort ellipse points
  vector<Point2f> outerElpsCntSubpx_und;
  undistortPoints(outerElpsCntSubpx, outerElpsCntSubpx_und, _cfg.K, _cfg.distortion, Mat(), _cfg.K);
  vector<Point2f> innerElpsCntSubpx_und;
  undistortPoints(innerElpsCntSubpx, innerElpsCntSubpx_und, _cfg.K, _cfg.distortion, Mat(), _cfg.K);

  // fit the ellipse
  Ellipse outerElpsSubpx = cv::fitEllipse(outerElpsCntSubpx_und);
  outerElps = outerElpsSubpx;
  Ellipse innerElpsSubpx = cv::fitEllipse(innerElpsCntSubpx_und);
  innerElps = innerElpsSubpx;

  // get ellipse polynomials
  EllipsePoly outerPoly, innerPoly;
  getEllipsePolynomialCoeff(outerElpsSubpx, outerPoly);
  getEllipsePolynomialCoeff(innerElpsSubpx, innerPoly);

  // compute center
  Point2d center;
  getDistanceWithGradientDescent(outerPoly, innerPoly, outerElps.center, 1e-6, 1e-2, center, 1e-8, 0);

  // compute distance
  getPoseGivenCenter(outerPoly, center, _cfg.markerDiameter / 2.0, tg->distance, tg->phi, tg->kappa);

  // get the center in the distorted image
  center = distort(center);
  tg->cx = center.x;
  tg->cy = center.y;

  tg->measured = true;

  return true;
}


/*
 * Compute min (white) and max (black) color from circle
 *
 */
void MarkerDetector_impl::evaluateExposure(const cv::Mat &raw, const Circle &outer, float heading, float &black, float &white) {

  Ellipse outerElps;
  fitEllipse(outer.cnt, outerElps);

  float outMargin = 10;
  float inMargin = 2;

  float outerElpsSize = (outerElps.size.width + outerElps.size.height) / 4.0;

  const float outScale = (outerElpsSize + outMargin) / outerElpsSize;
  const float inScale = (_cfg.markerDiameter + _cfg.markerInnerDiameter) / 2.0  / _cfg.markerDiameter;

  // TODO: should be at the correct percentage
  float safetyAngle = atan(inMargin / (outerElpsSize * inScale)); // to avoid getting the white dots

  float inc;
  vector<float> signal;

  getSignalInsideEllipse(raw, outerElps, outScale, signal, inc);

  white = 0;
  for (const float &d : signal) {
    white += d;
  }
  white /= signal.size();

  signal.clear();

  // TODO: check minimum and maximun angles, they are somewhat wrong
  vector<Point2f> pts;
  getSignalInsideEllipse(raw, outerElps, inScale, signal, inc,
      heading + 2.0 * M_PI * _cfg.markerSignalModel[4] + safetyAngle,
      heading + 2.0 * M_PI * _cfg.markerSignalModel[5] - safetyAngle, &pts);

  black = 0;
  for (const float &d : signal) {
    black += d;
  }
  black /= signal.size();

}


/*
 * Edge detection is done with Canny from openCV
 *
 */
void MarkerDetector_impl::detectEdges(const Mat& raw, Mat& edges) {

  Mat tmp;

  // with canny
  if (_cfg.CannyBlurKernelSize > 0) {
    blur(raw, tmp, Size(_cfg.CannyBlurKernelSize, _cfg.CannyBlurKernelSize));
    Canny(tmp, edges, _cfg.CannyLowerThreshold, _cfg.CannyHigherThreshold, 3, true);
  } else {
    Canny(raw, edges, _cfg.CannyLowerThreshold, _cfg.CannyHigherThreshold, 3, true);
  }
}


/*
 * Contours detection is done with openCV method
 *
 */
void MarkerDetector_impl::detectContours(const Mat &edges, Contours &ctrs) {
  ctrs.clear();
  findContours(edges, ctrs, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));
}

/*
 * This method filters coutours who looks like circles (approx. round)
 *
 */
void MarkerDetector_impl::filterContours(const Contours& in, Circles& out) {

  out.clear();

  for (auto cit = in.begin(); cit != in.end(); ++cit) {
    const Contour &c = *cit;

    if (c.size() < _cfg.contourFilterMinSize) {
      continue;
    }

    // compute center
    float cx = 0, cy = 0;
    for (auto it = c.begin(); it != c.end(); ++it) {
      cx += it->x;
      cy += it->y;
    }

    cx /= c.size();
    cy /= c.size();

    // compute average and std of distance
    float sumd = 0, sumd2 = 0, d2;
    for (auto it = c.begin(); it != c.end(); ++it) {
      d2 = pow(it->x - cx, 2) + pow(it->y - cy, 2);
      sumd2 += d2;
      sumd += sqrt(d2);
    }

    float rMean = sumd / c.size();
    float rStd = sqrt(
        c.size() * (sumd2 / c.size() - pow(rMean, 2)) / (c.size() - 1));

    // Attempt to have a metric that adapts to depth
    if ((rStd / rMean) < 0.075) {
      Point2f center(cx, cy);
      out.emplace_back(c, center, rMean);
    }

  }
}


/*
 * This method associate concentric circles into clusters
 * Best match is computed by center proximity and circle radius ratio (as declared in config)
 *
 */
void MarkerDetector_impl::clusterCircles(const Circles &in, std::vector<CircleCluster> &out) {

  // for each circle, do a cluster with the best matching inner circle, if found

  float radiusRatio = _cfg.markerInnerDiameter / _cfg.markerDiameter;

  for (int i = 0; i < in.size(); ++i) {

    int bestMatch = -1;
    float bestDiff = numeric_limits<float>::infinity();

    for (int j = 0; j < in.size(); ++j) {
      if (in[i].r > in[j].r
          && norm(in[i].center - in[j].center) < in[i].r * 0.5) {
        float curDiff = fabs(in[i].r * radiusRatio - in[j].r);

        if (curDiff < bestDiff) {
          bestMatch = j;
          bestDiff = curDiff;
        }
      }
    }

    if (bestMatch != -1) {
      if (fabs(in[bestMatch].r / in[i].r / radiusRatio - 1) < 0.25) {
        out.resize(out.size() + 1);
        out.back().center = in[i].center;
        out.back().rep = i;
        out.back().circleIds.push_back(i);
        out.back().circleIds.push_back(bestMatch);
      }
    }
  }

}

/*
 * For each cluster, this method compares signal with reference signal
 * Cluster having maximal correlation score selected and theta value (angle between ref and signal is computed)
 * returned values are selectedCluster and theta pointers
 *
 */
bool MarkerDetector_impl::selectMarker(const Mat& image, const Circles &candidates, const vector<int> &representativesIds, int &selectedCluster, float & theta) {

  bool found = false;

  float maxCorr = _cfg.markerxCorrThreshold;

  for (auto it = 0; it < representativesIds.size(); ++it) {

    const Circle &c = candidates[representativesIds[it]];

    vector<float> signal;
    getSignalInsideCircle(image, c, _cfg.markerSignalRadiusPercentage, signal);

    normalizeSignal(signal);

    Mat corr;
    computeNormalizedxCorr(signal, corr);

    double m, M;
    Point2i mLoc, MLoc;

    minMaxLoc(corr, &m, &M, &mLoc, &MLoc);

    if (M > maxCorr) {
      maxCorr = M;
      // compute theta
      theta = +M_PI - (float) MLoc.x / signal.size() * 2.0 * M_PI;

      found = true;
      selectedCluster = it;
    }
  }

  return found;
}

/*
 * Calling openCV to get an ellipse from contours
 */
void MarkerDetector_impl::fitEllipse(const Contour& cnt, Ellipse& out) {
  out = cv::fitEllipse(cnt);
}

/*
 * Normalize signal between -1 and 1 after computing signal min and max value
 *
 */
void MarkerDetector_impl::normalizeSignal(std::vector<float>& sig_in) {
  // depolarize
  Mat sig(1, sig_in.size(), CV_32FC1, sig_in.data());

  float u = mean(sig)[0];

  sig -= u;

  float M = -std::numeric_limits<float>::infinity();
  float m = std::numeric_limits<float>::infinity();

  for (unsigned int k = 0; k < sig_in.size(); k++) {
    if (sig_in[k] > M) {
      M = sig_in[k];
    } else if (sig_in[k] < m) {
      m = sig_in[k];
    }
  }

  if (m != M) {
    for (unsigned int k = 0; k < sig_in.size(); k++) {
      sig_in[k] = -1.0 + (sig_in[k] - m) * 2.0 / (M - m);
    }
  }
}

/*
 * computeNormalizedxCorr compares reference signal with picture's target signal
 * Give back out matrice with diff between ref and signal.
 */
void MarkerDetector_impl::computeNormalizedxCorr(const std::vector<float>& sig_in, Mat &out) {

  // prepare signals
  float raw[2 * sig_in.size()];

  for (unsigned int n = 0; n < 2; n++) {
    unsigned int segment = 0;
    float val = _cfg.markerSignalStartsWith;

    for (unsigned int k = 0; k < sig_in.size(); k++) {
      if (segment < _cfg.markerSignalModel.size()) {
        // test if I have to advance
        if (k > _cfg.markerSignalModel[segment] * sig_in.size()) {
          segment++;
          val = -val;
        }
      }
      raw[k + n * sig_in.size()] = val;
    }
  }

  Mat ref(1, 2 * sig_in.size(), CV_32FC1, raw);
  Mat sig(1, sig_in.size(), CV_32FC1, const_cast<float *>(sig_in.data()));

  // compute cross correlation
  matchTemplate(ref, sig, out, CV_TM_CCORR_NORMED);
}


/*
 * Get signal along a circle (given the circle and signal radius)
 *
 */
void MarkerDetector_impl::getSignalInsideCircle(const Mat& image, const Circle& circle, float radiusPercentage, std::vector<float>& signal) {

  int N = ceil(2 * M_PI / (1.0 / circle.r));

  signal.clear();
  signal.reserve(N);

  Mat px;
  float theta = -M_PI;

  for (int i = 0; i < N; ++i) {
    unsigned int x = round(circle.center.x + cos(theta) * circle.r * radiusPercentage);
    unsigned int y = round(circle.center.y + sin(theta) * circle.r * radiusPercentage);

    Scalar intensity = image.at<uchar>(y, x);
    signal.push_back(intensity[0]);

    theta += (2.0 * M_PI) / N;
  }
}


/*
 * Get signal along an ellipse (given ellipse)
 *
 */
void MarkerDetector_impl::getSignalInsideEllipse(const Mat& image, const Ellipse& ellipse, float radiusPercentage, std::vector<float>& signal,
                                                float &increment, float thetasmall, float thetabig, vector<Point2f> *pts) {

  increment = 1.0 / ((ellipse.size.width / 2.0 + ellipse.size.height / 2.0) / 2.0);

  int N = ceil((thetabig - thetasmall) / increment);

  signal.clear();
  signal.reserve(N);

  if (pts != NULL) {
    pts->clear();
    pts->reserve(N);
  }

  Mat px;
  float theta = thetasmall;

  for (int i = 0; i < N; ++i, theta += increment) {
    Point2f px = evalEllipse(theta, ellipse.center,
        ellipse.size.width * radiusPercentage / 2.0,
        ellipse.size.height * radiusPercentage / 2.0,
        ellipse.angle * M_PI / 180.0);

    Scalar intensity = image.at<uchar>(px.y, px.x);
    signal.push_back(intensity[0]);

    if (pts != NULL) {
      pts->push_back(px);
    }
  }
}


/*
 * Get X/Y coordinate of a point on an ellipse (determined by center, a & b radiuses) at a specified phi angle
 *
 */
Point2f MarkerDetector_impl::evalEllipse(float at, const Point2f& c, float a, float b, float phi) {

  float offset = 0.0;
  if (a < b) {
    offset = M_PI / 2.0;
    std::swap(a, b);
  }

  Point2f ret;

  ret.x = c.x + a * cos(at - phi + offset - M_PI) * cos(phi + offset) - b * sin(at - phi + offset - M_PI) * sin(phi + offset);
  ret.y = c.y + a * cos(at - phi + offset - M_PI) * sin(phi + offset) + b * sin(at - phi + offset - M_PI) * cos(phi + offset);

  return ret;
}

/*
 * Compute distance between Camera and target
 *
 */
void MarkerDetector_impl::getDistanceGivenCenter(const EllipsePoly& elps, const cv::Point2d& c, double r, double &mu, double &std, int N) {

  Point2d p1, p2;
  Eigen::Vector3d vr1, vr2, vrc;

  double sum = 0;
  double sumsq = 0;

  double f = _cfg.K.at<double>(0, 0); // focal lenght
  double cx = _cfg.K.at<double>(0, 2); // principal point x in pixels
  double cy = _cfg.K.at<double>(1, 2); // .. and y

  for (int i = 0; i < N; ++i) {
    double theta = (double) i / N * M_PI;

    getEllipseLineIntersections(elps, c.x, c.y, theta, p1, p2);

    // viewing rays
    vr1 << (p1.x - cx) / f, (p1.y - cy) / f, 1;
    vr2 << (p2.x - cx) / f, (p2.y - cy) / f, 1;
    vrc << (c.x - cx) / f, (c.y - cy) / f, 1;

    // angles

    double th1, th2;

    th1 = acos((vr1 / vr1.norm()).dot(vrc / vrc.norm()));
    th2 = acos((vr2 / vr2.norm()).dot(vrc / vrc.norm()));

    double curd = (sqrt(2) * r * sin(th1 + th2)) / sqrt(3 - 2 * cos(2 * th1) - 2 * cos(2 * th2) + cos(2 * (th1 + th2)));

    sum += curd;
    sumsq += pow(curd, 2);
  }

  mu = sum / N;
  std = sqrt(sumsq / N - pow(mu, 2));

  if (std::isnan(std)) {
    std = 0.0;
  }
}

/*
 * Computes camera angular position from camera parameters and target rotation
 * (I think we don't need this)
 */
void MarkerDetector_impl::getPoseGivenCenter(const EllipsePoly& elps, const cv::Point2d& c, double r, double &d, double &phi, double &kappa) {
  Point2d p1, p2;
  Eigen::Vector3d vr1, vr2, vrc;

  double f = _cfg.K.at<double>(0, 0); // focal lenght
  double cx = _cfg.K.at<double>(0, 2); // principal point x in pixels
  double cy = _cfg.K.at<double>(1, 2); // .. and y

  double theta;

  double th1, th2;

  Eigen::Array4d G1, B2, res; // candidate solutions
  int i;
  double ad, gamma1, beta2; // final solutions

  Eigen::Vector3d CB_phi, CB_kappa;
  double OB;

  // AROUND X AXIS
  theta = M_PI / 2;

  getEllipseLineIntersections(elps, c.x, c.y, theta, p1, p2);

  if (p2.y < p1.y) {
    swap(p1, p2);
  }

  // viewing rays
  vr1 << (p1.x - cx) / f, (p1.y - cy) / f, 1;
  vr2 << (p2.x - cx) / f, (p2.y - cy) / f, 1;
  vrc << (c.x - cx) / f, (c.y - cy) / f, 1;

  th1 = acos((vr1 / vr1.norm()).dot(vrc / vrc.norm()));
  th2 = acos((vr2 / vr2.norm()).dot(vrc / vrc.norm()));

  d = (sqrt(2) * r * sin(th1 + th2)) / sqrt(3 - 2 * cos(2 * th1) - 2 * cos(2 * th2) + cos(2 * (th1 + th2)));

  G1(0) = G1(1) = asin(d / r * sin(th1));
  G1(2) = G1(3) = M_PI - G1(0);

  B2(0) = B2(2) = asin(d / r * sin(th2));
  B2(1) = B2(3) = M_PI - B2(0);

  res = G1 + B2 + th1 + th2 - M_PI;
  ad = res.abs().minCoeff(&i);

  if (ad > 1e-6) {
    cerr << "WARNING, geometrical inconsistence: " << ad << endl;
  }

  gamma1 = G1(i);
  beta2 = B2(i);

  OB = r * sin(M_PI - beta2 - th2) / sin(th2);
  CB_phi = OB * vr2 - d * vrc;
  phi = atan2(CB_phi(1), CB_phi(2));

  // AROUND Y AXIS
  theta = 0.0;

  getEllipseLineIntersections(elps, c.x, c.y, theta, p1, p2);

  if (p2.x < p1.x) {
    swap(p1, p2);
  }

  // viewing rays
  vr1 << (p1.x - cx) / f, (p1.y - cy) / f, 1;
  vr2 << (p2.x - cx) / f, (p2.y - cy) / f, 1;
  vrc << (c.x - cx) / f, (c.y - cy) / f, 1;

  th1 = acos((vr1 / vr1.norm()).dot(vrc / vrc.norm()));
  th2 = acos((vr2 / vr2.norm()).dot(vrc / vrc.norm()));

  d = (sqrt(2) * r * sin(th1 + th2)) / sqrt(3 - 2 * cos(2 * th1) - 2 * cos(2 * th2) + cos(2 * (th1 + th2)));

  G1(0) = G1(1) = asin(d / r * sin(th1));
  G1(2) = G1(3) = M_PI - G1(0);

  B2(0) = B2(2) = asin(d / r * sin(th2));
  B2(1) = B2(3) = M_PI - B2(0);

  res = G1 + B2 + th1 + th2 - M_PI;
  ad = res.abs().minCoeff(&i);

  if (ad > 1e-6) {
    cerr << "WARNING, geometrical inconsistence: " << ad << endl;
  }

  gamma1 = G1(i);
  beta2 = B2(i);

  OB = r * sin(M_PI - beta2 - th2) / sin(th2);
  CB_kappa = OB * vr2 - d * vrc;
  kappa = atan2(CB_kappa(0), CB_kappa(2));
}


double MarkerDetector_impl::evalDistanceF(const EllipsePoly &outer, const EllipsePoly &inner, const cv::Point2d &x, const cv::Point2d &x0) {

  double mu, std;
  double ret;

  getDistanceGivenCenter(outer, x, _cfg.markerDiameter / 2.0, mu, std, 4);
  ret = std;
  ret += 0.1 * (pow(x.x - x0.x, 2) + pow(x.y - x0.y, 2));

  return ret;
}

void MarkerDetector_impl::getDistanceWithGradientDescent(const EllipsePoly& outer, const EllipsePoly& inner, const cv::Point2d x0, double step, 
                                                         double lambda, cv::Point2d& x, double tolX, double tolFun) {

  Eigen::VectorXd D;
  Point2d newx;

  x = x0;
  int it = 1;
  double f, newf;

  while (true) {

    // estimate gradient
    double fplus, fminus;
    Point2d g;

    // .. along x
    fplus = evalDistanceF(outer, inner, x + Point2d(step, 0.0), x0);
    fminus = evalDistanceF(outer, inner, x + Point2d(-step, 0.0), x0);
    g.x = (fplus - fminus) / 2.0 / step;

    // and along y
    fplus = evalDistanceF(outer, inner, x + Point2d(0.0, step), x0);
    fminus = evalDistanceF(outer, inner, x + Point2d(0.0, -step), x0);
    g.y = (fplus - fminus) / 2.0 / step;

    bool hadToReduce = false, stepDone = false;

    f = evalDistanceF(outer, inner, x, x0);

    while (lambda * norm(g) > tolX && !stepDone) {

      newx = x - lambda * g;
      newf = evalDistanceF(outer, inner, newx, x0);

      if (newf < f) {
        x = newx;
        f = newf;

        if (!hadToReduce) {
          lambda = lambda * 2.0;
        }

        stepDone = true;

//        cerr << it << " " << x << " " << newf << " " << lambda << endl;
      } else {
        lambda = lambda / 2.0;
        hadToReduce = true;
      }
    }

    if (!stepDone) {
      break;
    }

    if (fabs(f - newf) < tolFun) {
      break;
    }

    ++it;
  }
}


/*
 * Subpixel edge detection
 *
 */ 
void MarkerDetector_impl::subpixelEdgeWithLeastSquares(const cv::Mat &image, const Ellipse &elps, const EllipsePoly &poly, float theta, 
                                                        float a, float b, cv::Point2f &subpixedge, int N) {

  // evaluate preliminary edge position
  Point2f e = evalEllipse(theta, elps.center, elps.size.width / 2.0,
      elps.size.height / 2.0, elps.angle * M_PI / 180.0);

  // evaluate gradient at edge position
  float g = atan2(2.0 * poly(2) * e.y + poly(1) * e.x + poly(4),
      2.0 * poly(0) * e.x + poly(1) * e.y + poly(3));

  // compute a shift with respect to the preliminary edge position based on max gradient pixel
  float old = getSubpix(image,
      Point2f(e.x + (float) (-N - 1) * cos(g),
          e.y + (float) (-N - 1) * sin(g))), cur;
  float maxDelta = 0, delta;
  int maxDeltaI;
  for (int i = -N; i <= N + 1; ++i) {
    cur = getSubpix(image,
        Point2f(e.x + (float) i * cos(g), e.y + (float) i * sin(g)));
    delta = fabs(cur - old);
    if (delta > maxDelta) {
      maxDelta = delta;
      maxDeltaI = i;
    }

    old = cur;
  }

  // get pixel values along the orthogonal line
  vector<Point2f> orth(2 * N + 1);
  vector<float> subpixvalues(2 * N + 1);

  for (int i = -N; i <= N; ++i) {
    orth[i + N].x = e.x + (float) (i + maxDeltaI) * cos(g);
    orth[i + N].y = e.y + (float) (i + maxDeltaI) * sin(g);

    subpixvalues[i + N] = getSubpix(image, orth[i + N]);
  }

  // swap them in case order is inverted
  float swapmult = 1.0;
  if (subpixvalues[0] > subpixvalues[2 * N]) {
    swapmult = -1.0;
    for (int i = 0; i <= N; ++i) {
      swap(subpixvalues[i], subpixvalues[2 * N - i]);
    }
  }

  // toggle local limits estimation if not provided
  if (a < 0) {
    a = getSubpix(image, Point2f(e.x + (float) (-(N + 1) + maxDeltaI) * cos(g), e.y + (float) (-(N + 1) + maxDeltaI) * sin(g)));

    b = getSubpix(image, Point2f(e.x + (float) (N + 1 + maxDeltaI) * cos(g),  e.y + (float) (N + 1 + maxDeltaI) * sin(g)));

    if (swapmult == -1.0) {
      swap(a, b);
    }

    b = b - a;
  }
  //

  // do the least square thing
  // TODO: sometimes the estimation fails (dx = nan)
  float mu = 0.25, logsigma = 0;

  auto Sign = [](float x) {return x>=0 ? 1.0: -1.0;};

  Eigen::MatrixXd H(2 * N + 1, 2);
  Eigen::VectorXd err(2 * N + 1);

  Eigen::VectorXd dx(2);

  int it = 0;

  do {
    it++;

    float sigma = exp(logsigma);

    for (int i = -N; i <= N; ++i) {
      float y = i;

      // error
      err(i + N) = (2.0 * a + b
          - b
              * sqrt(
                  1.0 - exp((-2.0 * pow(mu - y, 2)) / (M_PI * pow(sigma, 2))))
              * Sign(mu - y)) / 2.0 - subpixvalues[i + N];

      // TODO: they could be optimized
      // derivative with respect to mu
      H(i + N, 0) = (b * (-mu + y) * Sign(mu - y)) / (exp((2.0 * pow(mu - y, 2)) / (M_PI * pow(sigma, 2))) 
                    * sqrt(1.0 - exp((-2.0 * pow(mu - y, 2)) / (M_PI * pow(sigma, 2))))  * M_PI * pow(sigma, 2));
      // derivative with respect to sigma
      H(i + N, 1) = sigma * (b * pow(mu - y, 2) * Sign(mu - y)) / (exp((2.0 * pow(mu - y, 2)) / (M_PI * pow(sigma, 2)))
              * sqrt(1.0 - exp((-2.0 * pow(mu - y, 2)) / (M_PI * pow(sigma, 2)))) * M_PI * pow(sigma, 3));
    }

    dx = (H.transpose() * H).inverse() * H.transpose() * err;

    mu = mu - dx(0);
    logsigma = logsigma - dx(1);

  } while (fabs(dx(0)) > 1e-2 && it <= 5);

  subpixedge = e  + Point2f((swapmult * mu + maxDeltaI) * cos(g), (swapmult * mu + maxDeltaI) * sin(g));
}

/*
 * Refines a ellipse contours with subpixel edges
 *
 */
void MarkerDetector_impl::refineEllipseCntWithSubpixelEdges(const cv::Mat &image, const Target &tg, const Ellipse &elps, bool ignoreSignalAreas, 
                                                            int N, std::vector<cv::Point2f> &cnt, std::vector<double> &angles) {

  EllipsePoly elpsPoly;
  getEllipsePolynomialCoeff(elps, elpsPoly);

  // decide the angular increment, avoid to reuse information
  float elpsSize = (elps.size.width + elps.size.height) / 4.0;
  float inc = 1.0 / (elpsSize - N);

  cnt.clear();
  cnt.reserve(ceil(2.0 * M_PI / inc));

  angles.clear();
  angles.reserve(ceil(2.0 * M_PI / inc));

  float theta = 0.0;
  float val = _cfg.markerSignalStartsWith;
  int segment = 0;
  float safetyAngle = 1 / elpsSize;

  while (theta < 2 * M_PI) {
    bool considerThisAngle = !ignoreSignalAreas;

    if (segment < _cfg.markerSignalModel.size() && theta > 2 * M_PI * _cfg.markerSignalModel[segment]) {
      ++segment;
      val = -val;
    }
    if (val == -1.0) {
      float minAngle = 2 * M_PI * _cfg.markerSignalModel[(segment - 1)  % _cfg.markerSignalModel.size()]  + safetyAngle;
      float maxAngle = 2 * M_PI * _cfg.markerSignalModel[(segment) % _cfg.markerSignalModel.size()] - safetyAngle
                       + (segment >= _cfg.markerSignalModel.size() ? 2 * M_PI : 0);

      if (theta >= minAngle && theta <= maxAngle) {
        considerThisAngle = true;
      }
    }

    if (considerThisAngle) {
      Point2f edge;

      // disable exposure hint
      subpixelEdgeWithLeastSquares(image, elps, elpsPoly, tg.heading + theta, -1, -1, edge, N);

      if (!isnan(edge.x) && !isnan(edge.y)) {
        cnt.push_back(edge);
        angles.push_back(tg.heading + theta);
      }
    }

    theta += inc;
  }
}

/*
 *
 * MWE : le floodfill est-il vraiment utile ?
 */
bool MarkerDetector_impl::measureRough(const cv::Mat &image, std::shared_ptr<Target> tg) {

  if (!tg->detected) {
    return false;
  }

  bool pointsFound = true;
  bool success = true;

  // check and eventually allocate mask
  if (_floodfillMask.rows != image.rows + 2
      || _floodfillMask.cols != image.cols + 2) {
    _floodfillMask = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
  }

  // color the mask so floodfill cannot surpass the outer circle
  for (auto it = tg->outer.cnt.begin(); it != tg->outer.cnt.end(); ++it) {
    _floodfillMask.at<unsigned char>(it->y + 1, it->x + 1) = 255;
  }

  // generate or validate seedpoints
  const unsigned int NPTS = _worldPoints.size();

  if (tg->seedPoints.empty()) {
    // get the seed points for the floodfill and the true world points
    Ellipse outerElps;
    fitEllipse(tg->outer.cnt, outerElps);

    for (int cnt = 0; cnt < _cfg.markerSignalModel.size() / 2; ++cnt) {
      int i = (_cfg.markerSignalStartsWith == 1.0 ? 0 : 1) + 2 * cnt;

      float maxAngle, minAngle, angle;

      if (i == 0) {
        minAngle = 2 * M_PI
            * (_cfg.markerSignalModel[_cfg.markerSignalModel.size() - 1] - 1);
      } else {
        minAngle = 2 * M_PI * _cfg.markerSignalModel[i - 1];
      }

      maxAngle = 2 * M_PI * _cfg.markerSignalModel[i];
      angle = 0.5 * (maxAngle + minAngle);

      tg->seedPoints.push_back(
          evalEllipse(angle + tg->heading, outerElps.center,
              outerElps.size.width / 2.0 * _cfg.markerSignalRadiusPercentage,
              outerElps.size.height / 2.0 * _cfg.markerSignalRadiusPercentage,
              outerElps.angle * M_PI / 180.0));
    }
  } else {
    if (tg->seedPoints.size() != NPTS) {
      cerr << "ERROR: not enough or too much seedpoints provided" << endl;
      assert(false);
    }
  }

  // floodfill and mask exploration to compute centroids
  Rect bounds[NPTS];
  int times = floor(253 / NPTS);

  for (int i = 0; i < NPTS; ++i) {
    floodFill(image, _floodfillMask, tg->seedPoints[i], 255, &(bounds[i]),
        (tg->white - tg->black) * 0.4,
        255,
        4 | ((2 + i * times) << 8) | CV_FLOODFILL_FIXED_RANGE
            | CV_FLOODFILL_MASK_ONLY);
  }

  unsigned int cnt[NPTS];
  for (unsigned int i = 0; i < NPTS; ++i) {
    cnt[i] = 0;
    tg->codePoints.push_back(Point2f(0.0, 0.0));
  }

  // compute centroids employing the bounds regions
  for (unsigned int i = 0; i < NPTS; ++i) {
    if (bounds[i].width == 0 || bounds[i].height == 0) {
      pointsFound = false; // I need to keep on to finish uncolouring the mask
    } else {
      for (unsigned int x = bounds[i].x; x <= bounds[i].x + bounds[i].width - 1;
          ++x) {
        for (unsigned int y = bounds[i].y;
            y <= bounds[i].y + bounds[i].height - 1; ++y) {
          unsigned char maskval = _floodfillMask.at<unsigned char>(y + 1,
              x + 1);

          if ((maskval - 2) / times == i) {
            // clear this mask point
            _floodfillMask.at<unsigned char>(y + 1, x + 1) = 0;

            // weighted average
            cnt[i] += image.at<unsigned char>(y, x);

            tg->codePoints[i].x += image.at<unsigned char>(y, x) * x;
            tg->codePoints[i].y += image.at<unsigned char>(y, x) * y;
          }
        }
      }
    }
  }

  // un-color the mask so floodfill cannot surpass the outer circle
  for (auto it = tg->outer.cnt.begin(); it != tg->outer.cnt.end(); ++it) {
    _floodfillMask.at<unsigned char>(it->y + 1, it->x + 1) = 0;
  }
  //*/

  vector<Point2f> prj_points;

  if (pointsFound) {
    for (unsigned int i = 0; i < NPTS; ++i) {
      tg->codePoints[i].x /= cnt[i];
      tg->codePoints[i].y /= cnt[i];
    }

    // solve the PnP problem
    Mat rod;
    solvePnP(_worldPoints, tg->codePoints, _cfg.K, _cfg.distortion, rod, tg->rought);
    Rodrigues(rod, tg->roughR);

    // reproject points and compute error
    projectPoints(_worldPoints, rod, tg->rought, _cfg.K, _cfg.distortion, prj_points);
    tg->meanReprojectionError = 0;

    for (unsigned int i = 0; i < NPTS; i++) {
      float err = sqrt(pow(prj_points[i].x - tg->codePoints[i].x, 2) + pow(prj_points[i].y - tg->codePoints[i].y, 2));

      if (err > 5) {
        cerr << "WARNING, high reprojection error " << i << "-th code point: "  << err << endl;
        success = false;
      }
      tg->meanReprojectionError += err;
    }

    tg->meanReprojectionError /= NPTS;
  } else {
    success = false;
  }

  if (success) {
    tg->roughlyMeasured = true;
  }

  return success;
}

/*
 * Apply camera lens distorting to position correctly a point as if lens was perfect.
 *
 */
Point2f MarkerDetector_impl::distort(const Point2f& p) {
  // To relative coordinates <- this is the step you are missing.
  double cx = _cfg.K.at<double>(0, 2);
  double cy = _cfg.K.at<double>(1, 2);
  double fx = _cfg.K.at<double>(0, 0);
  double fy = _cfg.K.at<double>(1, 1);
  double k1 = _cfg.distortion.at<double>(0);
  double k2 = _cfg.distortion.at<double>(1);
  double p1 = _cfg.distortion.at<double>(2);
  double p2 = _cfg.distortion.at<double>(3);
  double k3 = _cfg.distortion.at<double>(4);

  double x = (p.x - cx) / fx;
  double y = (p.y - cy) / fy;

  double r2 = x * x + y * y;

  // Radial distorsion
  double xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
  double yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

  // Tangential distorsion
  xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
  yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

  // Back to absolute coordinates.
  xDistort = xDistort * fx + cx;
  yDistort = yDistort * fy + cy;

  return Point2f(xDistort, yDistort);
}

/*
 * Computes polynomial parameters defining an ellipse
 *
 */
void MarkerDetector_impl::getEllipsePolynomialCoeff(const Ellipse &elps, EllipsePoly &poly) {
  double a, b, cx, cy, theta;

  a = elps.size.width / 2;
  b = elps.size.height / 2;
  theta = elps.angle * M_PI / 180.0;

  cx = elps.center.x;
  cy = elps.center.y;

  if (a < b) {
    theta += M_PI / 2.0;
    std::swap(a, b);
  }

  // Compute elements of conic matrix
  double A, B, C, D, E, F;

  A = pow(a, 2) * pow(sin(theta), 2) + pow(b, 2) * pow(cos(theta), 2);
  B = 2 * (pow(b, 2) - pow(a, 2)) * sin(theta) * cos(theta);
  C = pow(a, 2) * pow(cos(theta), 2) + pow(b, 2) * pow(sin(theta), 2);
  D = -2 * A * cx - B * cy;
  E = -B * cx - 2 * C * cy;
  F = A * pow(cx, 2) + B * cx * cy + C * pow(cy, 2) - pow(a, 2) * pow(b, 2);

  double k = 1.0 / sqrt((pow(A, 2) + pow(B, 2) + pow(C, 2) + pow(D, 2) + pow(E, 2) + pow(F, 2)));

  poly << A, B, C, D, E, F;
  poly *= k;

}

/*
 * Compute intersection points between a ellipse defined by its polynomial parameters 
 * and a line defined by a center(x0/y0) and an angle.
 *
 * Points are set in p1 and p2 pointers
 */
void MarkerDetector_impl::getEllipseLineIntersections(const EllipsePoly& em, double x0, double y0, double theta, cv::Point2d& p1, cv::Point2d& p2) {
  Eigen::Vector3d lm;

  if (fabs(theta - M_PI / 2.0) < 1e-6) {
    p1.x = x0;
    p1.y = (-(x0 * em(1)) - em(4) - sqrt(pow(x0 * em(1) + em(4), 2) - 4 * em(2) * (pow(x0, 2) * em(0) + x0 * em(3) + em(5)))) / (2 * em(2));
    p2.x = x0;
    p2.y = (-(x0 * em(1)) - em(4) + sqrt(pow(x0 * em(1) + em(4), 2) - 4 * em(2) * (pow(x0, 2) * em(0) + x0 * em(3) + em(5)))) / (2 * em(2));
  } else {
    lm << -tan(theta), 1.0, tan(theta) * x0 - y0;
  }
}
}
