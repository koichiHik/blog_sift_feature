/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
 Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

// STL
#include <iomanip>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Glog
#include <glog/logging.h>

// Original
#include "xfeatures2d_copy.hpp"

using namespace cv;

namespace cv_copy {

/******************************* Defs and macros *****************************/

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;

static Mat CreateInitialImage(const Mat& img, bool doubleImageSize, float sigma) {
  Mat gray, gray_fpt;
  if (img.channels() == 3 || img.channels() == 4) {
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
  } else {
    img.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
  }

  float sig_diff;

  if (doubleImageSize) {
    sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
    Mat dbl;
    resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR);
    GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
    return dbl;
  } else {
    sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
    GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
    return gray_fpt;
  }
}

static void DisplayImage(const cv::Mat& img, const std::string& title, const int wait_time) {
  double min_val, max_val;
  cv::minMaxLoc(img, &min_val, &max_val);
  cv::Mat tmp;

  img.convertTo(tmp, CV_8UC1, 255.0 / (max_val - min_val), -255.0 * min_val / (max_val - min_val));
  cv::minMaxLoc(tmp, &min_val, &max_val);
  cv::imshow(title, tmp);
  cv::waitKey(wait_time);
}

struct KeyPoint12_LessThan {
  bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const {
    if (kp1.pt.x != kp2.pt.x) return kp1.pt.x < kp2.pt.x;
    if (kp1.pt.y != kp2.pt.y) return kp1.pt.y < kp2.pt.y;
    if (kp1.size != kp2.size) return kp1.size > kp2.size;
    if (kp1.angle != kp2.angle) return kp1.angle < kp2.angle;
    if (kp1.response != kp2.response) return kp1.response > kp2.response;
    if (kp1.octave != kp2.octave) return kp1.octave > kp2.octave;
    return kp1.class_id > kp2.class_id;
  }
};

template <typename T>
static void RemoveDuplicateSorted(std::vector<T>& keypoints) {
  int i, j, n = (int)keypoints.size();

  if (n < 2) {
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(), KeyPoint12_LessThan());

  for (i = 0, j = 1; j < n; ++j) {
    const T& kp1 = keypoints[i];
    const T& kp2 = keypoints[j];

    // If two keypoints have same x, y, size and angle, exclude.
    if (kp1.pt.x != kp2.pt.x || kp1.pt.y != kp2.pt.y || kp1.size != kp2.size ||
        kp1.angle != kp2.angle) {
      keypoints[++i] = keypoints[j];
    }
  }
  keypoints.resize(i + 1);
}

struct KeypointResponseGreaterThanThreshold {
  KeypointResponseGreaterThanThreshold(float _value) : value(_value) {}
  inline bool operator()(const KeyPoint& kpt) const { return kpt.response >= value; }
  float value;
};

struct KeypointResponseGreater {
  inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const {
    return kp1.response > kp2.response;
  }
};

// Takes keypoints and culls them by the responce.
template <typename T>
void RetainBest(std::vector<T>& keypoints, int n_points) {
  // If the current keypoints number is below n_points, do nothing.
  if (n_points >= 0 && keypoints.size() > (size_t)n_points) {
    if (n_points == 0) {
      keypoints.clear();
      return;
    }
    // First use nth element to partition the keypoints into the best and worst.
    std::nth_element(keypoints.begin(), keypoints.begin() + n_points - 1, keypoints.end(),
                     KeypointResponseGreater());
    // This is the boundary response, and in the case of FAST may be ambiguous
    float ambiguous_response = keypoints[n_points - 1].response;
    // Use std::partition to grab all of the keypoints with the boundary response.
    typename std::vector<T>::const_iterator new_end =
        std::partition(keypoints.begin() + n_points, keypoints.end(),
                       KeypointResponseGreaterThanThreshold(ambiguous_response));
    // resize the keypoints, given this new end point. nth_element and partition reordered the
    // points inplace
    keypoints.resize(new_end - keypoints.begin());
  }
}

class BuildDiffOfGaussianPyramid : public ParallelLoopBody {
 public:
  BuildDiffOfGaussianPyramid(int num_gaussian_in_octave, int num_diff_of_gaussian_in_octave,
                             const std::vector<Mat>& gaussian_pyramid,
                             std::vector<Mat>& diff_of_gaussian_pyramid)
      : num_gaussian_in_octave_(num_gaussian_in_octave),
        num_diff_of_gaussian_in_octave_(num_diff_of_gaussian_in_octave),
        gaussian_pyramid_(gaussian_pyramid),
        diff_of_gaussian_pyramid_(diff_of_gaussian_pyramid) {}

  // Body of computation of DoG.
  void operator()(const cv::Range& range) const {
    const int begin = range.start;
    const int end = range.end;

    for (int a = begin; a < end; a++) {
      const int o = a / (num_diff_of_gaussian_in_octave_);
      const int i = a % (num_diff_of_gaussian_in_octave_);
      const Mat& src1 = gaussian_pyramid_[o * (num_gaussian_in_octave_) + i];
      const Mat& src2 = gaussian_pyramid_[o * (num_gaussian_in_octave_) + i + 1];
      Mat& dst = diff_of_gaussian_pyramid_[o * (num_diff_of_gaussian_in_octave_) + i];
      subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
    }
  }

 private:
  int num_gaussian_in_octave_;
  int num_diff_of_gaussian_in_octave_;
  const std::vector<Mat>& gaussian_pyramid_;
  std::vector<Mat>& diff_of_gaussian_pyramid_;
};

// Computes a gradient orientation histogram at a specified pixel
static float ComputeOrientationHistogram(const Mat& img, const Point2i& pt, int radius, float sigma,
                                         float* hist, int num_hist_bins) {
  int len = (radius * 2 + 1) * (radius * 2 + 1);
  float expf_scale = -1.f / (2.f * sigma * sigma);

  // Adjusting memory space.
  AutoBuffer<float> buf(len * 4 + num_hist_bins + 4);
  // Gradient (Temporary.)
  float* X = buf.data();
  float* Y = X + len;

  // Magnitude, orientation, weighted magnitude. (Final output.)
  float* Mag = X;
  float* Ori = Y + len;
  float* W = Ori + len;

  // Temporary histogram storage.
  float* temphist = W + len + 2;

  for (int bin = 0; bin < num_hist_bins; bin++) {
    temphist[bin] = 0.f;
  }

  // Loop for square that encircles keypoint.
  int k = 0;

  // Compute gradient and magnitude.
  {
    for (int dy = -radius; dy <= radius; dy++) {
      int y = pt.y + dy;
      // Additional 1pix is necessary for difference.
      if (y < 1 || img.rows - 2 < y) {
        continue;
      }
      for (int dx = -radius; dx <= radius; dx++) {
        int x = pt.x + dx;
        // Additional 1pix is necessary for difference.
        if (x < 1 || img.cols - 2 < x) {
          continue;
        }

        // Gradient.
        X[k] = (float)(img.at<sift_wt>(y, x + 1) - img.at<sift_wt>(y, x - 1));
        Y[k] = (float)(img.at<sift_wt>(y - 1, x) - img.at<sift_wt>(y + 1, x));
        W[k] = (dx * dx + dy * dy) * expf_scale;
        k++;
      }
    }
  }
  len = k;

  // Orientation Histogram Computation.
  float max_val;
  {
    // Compute gradient values, orientations and the weights over the pixel neighborhood.
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    k = 0;
    for (; k < len; k++) {
      int bin = cvRound((num_hist_bins / 360.f) * Ori[k]);
      if (bin >= num_hist_bins) {
        bin -= num_hist_bins;
      }
      if (bin < 0) {
        bin += num_hist_bins;
      }
      temphist[bin] += W[k] * Mag[k];
    }

    // Smooth the histogram.
    temphist[-1] = temphist[num_hist_bins - 1];
    temphist[-2] = temphist[num_hist_bins - 2];
    temphist[num_hist_bins] = temphist[0];
    temphist[num_hist_bins + 1] = temphist[1];

    for (int bins = 0; bins < num_hist_bins; bins++) {
      hist[bins] = (temphist[bins - 2] + temphist[bins + 2]) * (1.f / 16.f) +
                   (temphist[bins - 1] + temphist[bins + 1]) * (4.f / 16.f) +
                   temphist[bins] * (6.f / 16.f);
    }

    max_val = hist[0];
    for (int bins = 1; bins < num_hist_bins; bins++) {
      max_val = std::max(max_val, hist[bins]);
    }
  }

  return max_val;
}

static Vec3f ComputeGradientOfDoG(const int row, const int col, const float deriv_scale,
                                  const Mat& prev_dog, const Mat& mid_dog, const Mat& next_dog) {
  Vec3f dD((mid_dog.at<sift_wt>(row, col + 1) - mid_dog.at<sift_wt>(row, col - 1)) * deriv_scale,
           (mid_dog.at<sift_wt>(row + 1, col) - mid_dog.at<sift_wt>(row - 1, col)) * deriv_scale,
           (next_dog.at<sift_wt>(row, col) - prev_dog.at<sift_wt>(row, col)) * deriv_scale);

  return dD;
}

static Matx33f ComputeHessianOfDoG(const int row, const int col, const float second_deriv_scale,
                                   const float cross_deriv_scale, const Mat& prev_dog,
                                   const Mat& mid_dog, const Mat& next_dog) {
  float v2 = (float)mid_dog.at<sift_wt>(row, col) * 2;
  float dxx = (mid_dog.at<sift_wt>(row, col + 1) + mid_dog.at<sift_wt>(row, col - 1) - v2) *
              second_deriv_scale;
  float dyy = (mid_dog.at<sift_wt>(row + 1, col) + mid_dog.at<sift_wt>(row - 1, col) - v2) *
              second_deriv_scale;
  float dss =
      (next_dog.at<sift_wt>(row, col) + prev_dog.at<sift_wt>(row, col) - v2) * second_deriv_scale;
  float dxy = (mid_dog.at<sift_wt>(row + 1, col + 1) - mid_dog.at<sift_wt>(row + 1, col - 1) -
               mid_dog.at<sift_wt>(row - 1, col + 1) + mid_dog.at<sift_wt>(row - 1, col - 1)) *
              cross_deriv_scale;
  float dxs = (next_dog.at<sift_wt>(row, col + 1) - next_dog.at<sift_wt>(row, col - 1) -
               prev_dog.at<sift_wt>(row, col + 1) + prev_dog.at<sift_wt>(row, col - 1)) *
              cross_deriv_scale;
  float dys = (next_dog.at<sift_wt>(row + 1, col) - next_dog.at<sift_wt>(row - 1, col) -
               prev_dog.at<sift_wt>(row + 1, col) + prev_dog.at<sift_wt>(row - 1, col)) *
              cross_deriv_scale;

  Matx33f H(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);

  return H;
}

static bool RefineKeyPointLocationIntoSubPixel(
    const int num_split_in_octave, const int octv, const float contrast_threshold,
    const float edge_threshold, const float sigma, const float deriv_scale,
    const float second_deriv_scale, const float cross_deriv_scale,
    const std::vector<Mat>& diff_of_gaussian_pyr, Vec3i& X, Vec3f& dX) {
  // Copy into modifiable buffer.

  for (int i = 0; i <= SIFT_MAX_INTERP_STEPS; i++) {
    if (i >= SIFT_MAX_INTERP_STEPS) {
      return false;
    }

    int idx = octv * (num_split_in_octave + 2) + X[2];
    const Mat& prev_dog = diff_of_gaussian_pyr[idx - 1];
    const Mat& mid_dog = diff_of_gaussian_pyr[idx];
    const Mat& next_dog = diff_of_gaussian_pyr[idx + 1];

    Vec3f dD = ComputeGradientOfDoG(X[1], X[0], deriv_scale, prev_dog, mid_dog, next_dog);
    Matx33f H = ComputeHessianOfDoG(X[1], X[0], second_deriv_scale, cross_deriv_scale, prev_dog,
                                    mid_dog, next_dog);
    dX = -H.solve(dD, DECOMP_LU);
    if (std::abs(dX[0]) < 0.5f && std::abs(dX[1]) < 0.5f && std::abs(dX[2]) < 0.5f) {
      break;
    }

    if (std::abs(dX[0]) > (float)(INT_MAX / 3) || std::abs(dX[1]) > (float)(INT_MAX / 3) ||
        std::abs(dX[2]) > (float)(INT_MAX / 3)) {
      return false;
    }

    X = X + Vec3i(cvRound(dX[0]), cvRound(dX[1]), cvRound(dX[2]));

    if (X[2] < 1 || X[2] > num_split_in_octave || X[0] < SIFT_IMG_BORDER ||
        X[0] >= mid_dog.cols - SIFT_IMG_BORDER || X[1] < SIFT_IMG_BORDER ||
        X[1] >= mid_dog.rows - SIFT_IMG_BORDER) {
      return false;
    }
  }

  return true;
}

static bool IsKeyPointWithEnoughContrastOnNonEdge(
    const Vec3i& X, const Vec3f& dX, const int octv, const int num_split_in_octave,
    const std::vector<Mat>& diff_of_gaussian_pyr, const float contrast_threshold,
    const float edge_threshold, const float img_scale, const float deriv_scale,
    const float second_deriv_scale, const float cross_deriv_scale, float& contrast) {
  int col = X[0];
  int row = X[1];
  int layer = X[2];

  int idx = octv * (num_split_in_octave + 2) + layer;
  const Mat& img = diff_of_gaussian_pyr[idx];
  const Mat& prev = diff_of_gaussian_pyr[idx - 1];
  const Mat& next = diff_of_gaussian_pyr[idx + 1];
  Matx31f dD((img.at<sift_wt>(row, col + 1) - img.at<sift_wt>(row, col - 1)) * deriv_scale,
             (img.at<sift_wt>(row + 1, col) - img.at<sift_wt>(row - 1, col)) * deriv_scale,
             (next.at<sift_wt>(row, col) - prev.at<sift_wt>(row, col)) * deriv_scale);
  float t = dD.dot(Matx31f(dX[0], dX[1], dX[2]));

  contrast = img.at<sift_wt>(row, col) * img_scale + t * 0.5f;
  if (std::abs(contrast) * num_split_in_octave < contrast_threshold) {
    return false;
  }

  // principal curvatures are computed using the trace and det of Hessian
  float v2 = img.at<sift_wt>(row, col) * 2.f;
  float dxx =
      (img.at<sift_wt>(row, col + 1) + img.at<sift_wt>(row, col - 1) - v2) * second_deriv_scale;
  float dyy =
      (img.at<sift_wt>(row + 1, col) + img.at<sift_wt>(row - 1, col) - v2) * second_deriv_scale;
  float dxy = (img.at<sift_wt>(row + 1, col + 1) - img.at<sift_wt>(row + 1, col - 1) -
               img.at<sift_wt>(row - 1, col + 1) + img.at<sift_wt>(row - 1, col - 1)) *
              cross_deriv_scale;
  float tr = dxx + dyy;
  float det = dxx * dyy - dxy * dxy;

  if (det <= 0 || tr * tr * edge_threshold >= (edge_threshold + 1) * (edge_threshold + 1) * det) {
    return false;
  }

  return true;
}

// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
static bool RefineLocalExtremaIntoSubPixel(const std::vector<Mat>& diff_of_gaussian_pyr,
                                           SiftKeyPoint& keypoint, int octv, int& layer, int& row,
                                           int& col, int num_split_in_octave,
                                           float contrast_threshold, float edge_threshold,
                                           float sigma) {
  const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
  const float deriv_scale = img_scale * 0.5f;
  const float second_deriv_scale = img_scale;
  const float cross_deriv_scale = img_scale * 0.25f;

  Vec3i X(col, row, layer);
  Vec3f dX;

  bool sub_pix_refine_success = RefineKeyPointLocationIntoSubPixel(
      num_split_in_octave, octv, contrast_threshold, edge_threshold, sigma, deriv_scale,
      second_deriv_scale, cross_deriv_scale, diff_of_gaussian_pyr, X, dX);

  if (!sub_pix_refine_success) {
    return false;
  }

  col = X[0];
  row = X[1];
  layer = X[2];

  float contrast;
  bool keypoint_is_sound = IsKeyPointWithEnoughContrastOnNonEdge(
      X, dX, octv, num_split_in_octave, diff_of_gaussian_pyr, contrast_threshold, edge_threshold,
      img_scale, deriv_scale, second_deriv_scale, cross_deriv_scale, contrast);

  if (!keypoint_is_sound) {
    return false;
  }

  keypoint.pt.x = (col + dX[0]) * (1 << octv);
  keypoint.pt.y = (row + dX[1]) * (1 << octv);
  keypoint.octave = octv;
  keypoint.layer = layer;

  keypoint.size = sigma * powf(2.f, (layer + dX[2]) / num_split_in_octave) * (1 << octv) * 2;
  keypoint.response = std::abs(contrast);

  return true;
}

template <typename T>
static T NormalizeValueForHistBins(const T value, const int num_hist_bins) {
  T normalized_value = value;
  // T casted_num_hist_bins = static_cast<T>(num_hist_bins);

  if (normalized_value < 0) {
    normalized_value = normalized_value + num_hist_bins;
  } else if (num_hist_bins <= normalized_value) {
    normalized_value = normalized_value - num_hist_bins;
  }
  return normalized_value;
}

static void ComputeOrientationForKeyPoint(const int oct_idx_, const int num_split_in_octave_,
                                          const int cand_row, const int cand_col, const int layer,
                                          const std::vector<Mat> gaussian_pyramid_,
                                          const int num_hist_bins, float* hist, SiftKeyPoint& kpt,
                                          std::vector<SiftKeyPoint>* tls_kpts) {
  // Compute orientation.
  float scl_octv = kpt.size * 0.5f / (1 << oct_idx_);

  float orientation_max_val = ComputeOrientationHistogram(
      gaussian_pyramid_[oct_idx_ * (num_split_in_octave_ + 3) + layer], Point2i(cand_col, cand_row),
      cvRound(SIFT_ORI_RADIUS * scl_octv), SIFT_ORI_SIG_FCTR * scl_octv, hist, num_hist_bins);

  // Compute threshold above which the orientation will be adopted.
  float orientation_threshold = (float)(orientation_max_val * SIFT_ORI_PEAK_RATIO);
  for (int cur_bin = 0; cur_bin < num_hist_bins; cur_bin++) {
    int left_bin = 0 < cur_bin ? cur_bin - 1 : num_hist_bins - 1;
    int right_bin = cur_bin < num_hist_bins - 1 ? cur_bin + 1 : 0;

    // Peak check.
    if (!(hist[left_bin] < hist[cur_bin] && hist[right_bin] < hist[cur_bin])) {
      continue;
    }
    // Threshold check.
    if (!(orientation_threshold <= hist[cur_bin])) {
      continue;
    }

    // Parabora fitting.
    float decimal_bin = cur_bin + 0.5f * (hist[left_bin] - hist[right_bin]) /
                                      (hist[left_bin] - 2 * hist[cur_bin] + hist[right_bin]);
    decimal_bin = NormalizeValueForHistBins(decimal_bin, num_hist_bins);
    // Coordinate transform and conversion from bin to angle.
    float bin_resolution = (float)(360.f / num_hist_bins);
    kpt.angle = 360.f - bin_resolution * decimal_bin;

    if (std::abs(kpt.angle - 360.f) < FLT_EPSILON) {
      kpt.angle = 0.f;
    }
    { tls_kpts->push_back(kpt); }
  }
}

class FindScaleSpaceExtremaComputer : public ParallelLoopBody {
 public:
  FindScaleSpaceExtremaComputer(int oct_idx, int split_idx, int threshold, int dog_idx, int stride,
                                int cols, int num_split_in_octave, double contrast_threshold,
                                double edge_threshold, double sigma,
                                const std::vector<Mat>& gaussian_pyramid,
                                const std::vector<Mat>& diff_of_gaussian_pyramid,
                                TLSData<std::vector<SiftKeyPoint> >& _tls_kpts_struct)

      : oct_idx_(oct_idx),
        split_idx_(split_idx),
        threshold_(threshold),
        dog_idx_(dog_idx),
        stride_(stride),
        cols_(cols),
        num_split_in_octave_(num_split_in_octave),
        contrast_threshold_(contrast_threshold),
        edge_threshold_(edge_threshold),
        sigma_(sigma),
        gaussian_pyramid_(gaussian_pyramid),
        diff_of_gaussian_pyramid_(diff_of_gaussian_pyramid),
        tls_kpts_struct(_tls_kpts_struct) {}

  void operator()(const cv::Range& range) const {
    const int begin = range.start;
    const int end = range.end;

    const Mat& prev = diff_of_gaussian_pyramid_[dog_idx_ - 1];
    const Mat& img = diff_of_gaussian_pyramid_[dog_idx_];
    const Mat& next = diff_of_gaussian_pyramid_[dog_idx_ + 1];

    std::vector<SiftKeyPoint>* tls_kpts = tls_kpts_struct.get();

    static const int num_hist_bins = SIFT_ORI_HIST_BINS;
    float hist[num_hist_bins];
    SiftKeyPoint kpt;
    // Outer loop is for row.
    for (int row = begin; row < end; row++) {
      const sift_wt* prevptr = prev.ptr<sift_wt>(row);
      const sift_wt* currptr = img.ptr<sift_wt>(row);
      const sift_wt* nextptr = next.ptr<sift_wt>(row);

      // Inner loop is for col.
      for (int col = SIFT_IMG_BORDER; col < cols_ - SIFT_IMG_BORDER; col++) {
        sift_wt val = currptr[col];

        // Find local extrema with pixel accuracy.
        if (std::abs(val) > threshold_ &&
            (IsPixelLocalMaxima(prevptr, currptr, nextptr, col, val) ||
             IsPixelLocalMinima(prevptr, currptr, nextptr, col, val))) {
          int cand_row = row;
          int cand_col = col;
          int layer = split_idx_;

          // Compute subpixel location of extrema.
          bool subpix_estimation_success = RefineLocalExtremaIntoSubPixel(
              diff_of_gaussian_pyramid_, kpt, oct_idx_, layer, cand_row, cand_col,
              num_split_in_octave_, (float)contrast_threshold_, (float)edge_threshold_,
              (float)sigma_);

          // If fail, go to next pix.
          if (!subpix_estimation_success) {
            continue;
          }

          // Compute orientation for key point.
          ComputeOrientationForKeyPoint(oct_idx_, num_split_in_octave_, cand_row, cand_col, layer,
                                        gaussian_pyramid_, num_hist_bins, hist, kpt, tls_kpts);
        }
      }
    }
  }

 private:
  bool IsPixelLocalMaxima(const sift_wt* prev_ptr, const sift_wt* cur_ptr, const sift_wt* next_ptr,
                          int col, sift_wt val) const {
    return (val > 0 && val >= cur_ptr[col - 1] && val >= cur_ptr[col + 1] &&
            val >= cur_ptr[col - stride_ - 1] && val >= cur_ptr[col - stride_] &&
            val >= cur_ptr[col - stride_ + 1] && val >= cur_ptr[col + stride_ - 1] &&
            val >= cur_ptr[col + stride_] && val >= cur_ptr[col + stride_ + 1] &&
            val >= next_ptr[col] && val >= next_ptr[col - 1] && val >= next_ptr[col + 1] &&
            val >= next_ptr[col - stride_ - 1] && val >= next_ptr[col - stride_] &&
            val >= next_ptr[col - stride_ + 1] && val >= next_ptr[col + stride_ - 1] &&
            val >= next_ptr[col + stride_] && val >= next_ptr[col + stride_ + 1] &&
            val >= prev_ptr[col] && val >= prev_ptr[col - 1] && val >= prev_ptr[col + 1] &&
            val >= prev_ptr[col - stride_ - 1] && val >= prev_ptr[col - stride_] &&
            val >= prev_ptr[col - stride_ + 1] && val >= prev_ptr[col + stride_ - 1] &&
            val >= prev_ptr[col + stride_] && val >= prev_ptr[col + stride_ + 1]);
  }

  bool IsPixelLocalMinima(const sift_wt* prev_ptr, const sift_wt* cur_ptr, const sift_wt* next_ptr,
                          int col, sift_wt val) const {
    return (val < 0 && val <= cur_ptr[col - 1] && val <= cur_ptr[col + 1] &&
            val <= cur_ptr[col - stride_ - 1] && val <= cur_ptr[col - stride_] &&
            val <= cur_ptr[col - stride_ + 1] && val <= cur_ptr[col + stride_ - 1] &&
            val <= cur_ptr[col + stride_] && val <= cur_ptr[col + stride_ + 1] &&
            val <= next_ptr[col] && val <= next_ptr[col - 1] && val <= next_ptr[col + 1] &&
            val <= next_ptr[col - stride_ - 1] && val <= next_ptr[col - stride_] &&
            val <= next_ptr[col - stride_ + 1] && val <= next_ptr[col + stride_ - 1] &&
            val <= next_ptr[col + stride_] && val <= next_ptr[col + stride_ + 1] &&
            val <= prev_ptr[col] && val <= prev_ptr[col - 1] && val <= prev_ptr[col + 1] &&
            val <= prev_ptr[col - stride_ - 1] && val <= prev_ptr[col - stride_] &&
            val <= prev_ptr[col - stride_ + 1] && val <= prev_ptr[col + stride_ - 1] &&
            val <= prev_ptr[col + stride_] && val <= prev_ptr[col + stride_ + 1]);
  }

 private:
  int oct_idx_, split_idx_, dog_idx_;
  int threshold_;
  int stride_, cols_;
  int num_split_in_octave_;
  double contrast_threshold_;
  double edge_threshold_;
  double sigma_;
  const std::vector<Mat>& gaussian_pyramid_;
  const std::vector<Mat>& diff_of_gaussian_pyramid_;
  TLSData<std::vector<SiftKeyPoint> >& tls_kpts_struct;
};

// Compute sift descriptor.
static void ComputeSIFTDescriptor(const Mat& img, Point2f ptf, float angle, float scl,
                                  int desc_patch_width, int desc_hist_bins, float* dst) {
  Point2i pt(cvRound(ptf.x), cvRound(ptf.y));
  float bin_resolution = desc_hist_bins / 360.f;
  float exp_scale = -1.f / (desc_patch_width * desc_patch_width * 0.5f);
  float hist_width = SIFT_DESCR_SCL_FCTR * scl;

  // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
  int radius;
  float *X, *Y, *Mag, *Ori, *W, *RBin, *CBin, *hist;
  AutoBuffer<float> buf;
  {
    float sqrt2 = 1.4142135623730951f;
    radius = cvRound(hist_width * sqrt2 * (desc_patch_width + 1) * 0.5f);
    radius =
        std::min(radius, (int)sqrt(((double)img.cols) * img.cols + ((double)img.rows) * img.rows));

    int pre_patch_size = (radius * 2 + 1) * (radius * 2 + 1);
    int desc_hist_size = (desc_patch_width + 2) * (desc_patch_width + 2) * (desc_hist_bins + 2);
    buf.allocate(pre_patch_size * 6 + desc_hist_size);

    // Elements to be calculated for each prepatch.
    X = buf.data();
    Y = X + pre_patch_size;
    Mag = Y;
    Ori = Mag + pre_patch_size;
    W = Ori + pre_patch_size;
    RBin = W + pre_patch_size;
    CBin = RBin + pre_patch_size;

    hist = CBin + pre_patch_size;

    // Initialize histogram part of the buffer..　
    for (int i = 0; i < desc_patch_width + 2; i++) {
      for (int j = 0; j < desc_patch_width + 2; j++) {
        for (int k = 0; k < desc_hist_bins + 2; k++) {
          hist[(i * (desc_patch_width + 2) + j) * (desc_hist_bins + 2) + k] = 0.;
        }
      }
    }
  }

  // Computation for pre-patch element.
  int pix_cnt = 0;
  {
    float cos_t = cosf(angle * (float)(CV_PI / 180));
    float sin_t = sinf(angle * (float)(CV_PI / 180));
    cos_t /= hist_width;
    sin_t /= hist_width;

    // Compute gradients and weighted gradients of pixels around keypoint.
    for (int dy = -radius; dy <= radius; dy++) {
      for (int dx = -radius; dx <= radius; dx++) {
        // Calculate sample's histogram array coords rotated relative to ori.
        // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
        // r_rot = 1.5) have full weight placed in row 1 after interpolation.

        // Rotate for keypoint orientation.
        float dx_rot = dx * cos_t - dy * sin_t;
        float dy_rot = dx * sin_t + dy * cos_t;

        // Transform to final coordinate.
        float cbin = dx_rot + desc_patch_width / 2 - 0.5f;
        float rbin = dy_rot + desc_patch_width / 2 - 0.5f;
        int row = pt.y + dy, col = pt.x + dx;

        if (-1 < rbin && rbin < desc_patch_width && -1 < cbin && cbin < desc_patch_width &&
            0 < row && row < img.rows - 1 && 0 < col && col < img.cols - 1) {
          X[pix_cnt] = (float)(img.at<sift_wt>(row, col + 1) - img.at<sift_wt>(row, col - 1));
          Y[pix_cnt] = (float)(img.at<sift_wt>(row - 1, col) - img.at<sift_wt>(row + 1, col));
          RBin[pix_cnt] = rbin;
          CBin[pix_cnt] = cbin;
          W[pix_cnt] = (dx_rot * dx_rot + dy_rot * dy_rot) * exp_scale;
          pix_cnt++;
        }
      }
    }
  }

  {
    cv::hal::fastAtan2(Y, X, Ori, pix_cnt, true);
    cv::hal::magnitude32f(X, Y, Mag, pix_cnt);
    cv::hal::exp32f(W, W, pix_cnt);

    for (int pix_idx = 0; pix_idx < pix_cnt; pix_idx++) {
      float rbin = RBin[pix_idx];
      float cbin = CBin[pix_idx];
      float obin = (Ori[pix_idx] - angle) * bin_resolution;
      float mag = Mag[pix_idx] * W[pix_idx];

      int r0 = cvFloor(rbin);
      int c0 = cvFloor(cbin);
      int o0 = cvFloor(obin);
      rbin -= r0;
      cbin -= c0;
      obin -= o0;

      if (o0 < 0) {
        o0 += desc_hist_bins;
      }
      if (o0 >= desc_hist_bins) {
        o0 -= desc_hist_bins;
      }

      // histogram update using tri-linear interpolation
      float v_r1 = mag * rbin;
      float v_r0 = mag - v_r1;
      float v_rc11 = v_r1 * cbin;
      float v_rc10 = v_r1 - v_rc11;
      float v_rc01 = v_r0 * cbin;
      float v_rc00 = v_r0 - v_rc01;
      float v_rco111 = v_rc11 * obin;
      float v_rco110 = v_rc11 - v_rco111;
      float v_rco101 = v_rc10 * obin;
      float v_rco100 = v_rc10 - v_rco101;
      float v_rco011 = v_rc01 * obin;
      float v_rco010 = v_rc01 - v_rco011;
      float v_rco001 = v_rc00 * obin;
      float v_rco000 = v_rc00 - v_rco001;

      int idx = ((r0 + 1) * (desc_patch_width + 2) + c0 + 1) * (desc_hist_bins + 2) + o0;
      hist[idx] += v_rco000;
      hist[idx + 1] += v_rco001;
      hist[idx + (desc_hist_bins + 2)] += v_rco010;
      hist[idx + (desc_hist_bins + 3)] += v_rco011;
      hist[idx + (desc_patch_width + 2) * (desc_hist_bins + 2)] += v_rco100;
      hist[idx + (desc_patch_width + 2) * (desc_hist_bins + 2) + 1] += v_rco101;
      hist[idx + (desc_patch_width + 3) * (desc_hist_bins + 2)] += v_rco110;
      hist[idx + (desc_patch_width + 3) * (desc_hist_bins + 2) + 1] += v_rco111;
    }
  }

  // finalize histogram, since the orientation histograms are circular
  for (int i = 0; i < desc_patch_width; i++) {
    for (int j = 0; j < desc_patch_width; j++) {
      int idx = ((i + 1) * (desc_patch_width + 2) + (j + 1)) * (desc_hist_bins + 2);
      hist[idx] += hist[idx + desc_hist_bins];
      hist[idx + 1] += hist[idx + desc_hist_bins + 1];
      for (int k = 0; k < desc_hist_bins; k++) {
        dst[(i * desc_patch_width + j) * desc_hist_bins + k] = hist[idx + k];
      }
    }
  }
  // copy histogram to the descriptor,
  // apply hysteresis thresholding
  // and scale the result, so that it can be easily converted
  // to byte array
  float nrm2 = 0;
  int len = desc_patch_width * desc_patch_width * desc_hist_bins;
  for (int k = 0; k < len; k++) {
    nrm2 += dst[k] * dst[k];
  }

  float thr = std::sqrt(nrm2) * SIFT_DESCR_MAG_THR;

  nrm2 = 0;
  for (int i = 0; i < len; i++) {
    float val = std::min(dst[i], thr);
    dst[i] = val;
    nrm2 += val * val;
  }
  nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

  for (int k = 0; k < len; k++) {
    dst[k] = saturate_cast<uchar>(dst[k] * nrm2);
  }
}  // namespace cv_copy

class ComputeDescriptorComputer : public ParallelLoopBody {
 public:
  ComputeDescriptorComputer(const std::vector<Mat>& _gpyr,
                            const std::vector<SiftKeyPoint>& _keypoints, Mat& _descriptors,
                            int num_split_in_octave, int _firstOctave)
      : gpyr(_gpyr),
        keypoints(_keypoints),
        descriptors(_descriptors),
        num_split_in_octave_(num_split_in_octave),
        firstOctave(_firstOctave) {}

  void operator()(const cv::Range& range) const {
    const int kpt_start = range.start;
    const int kpt_end = range.end;

    static const int desc_patch_width = SIFT_DESCR_WIDTH;
    static const int desc_hist_bins = SIFT_DESCR_HIST_BINS;

    for (int idx = kpt_start; idx < kpt_end; idx++) {
      SiftKeyPoint kpt = keypoints[idx];
      float scale = std::pow(2.0, -kpt.octave);
      float size = kpt.size * scale;
      Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);
      const Mat& img = gpyr[(kpt.octave - firstOctave) * (num_split_in_octave_ + 3) + kpt.layer];
      float angle = 360.f - kpt.angle;
      if (std::abs(angle - 360.f) < FLT_EPSILON) {
        angle = 0.f;
      }
      float* p_desc = descriptors.ptr<float>((int)idx);
      ComputeSIFTDescriptor(img, ptf, angle, size * 0.5f, desc_patch_width, desc_hist_bins, p_desc);
    }
  }

 private:
  const std::vector<Mat>& gpyr;
  const std::vector<SiftKeyPoint>& keypoints;
  Mat& descriptors;
  int num_split_in_octave_;
  int firstOctave;
};

static void ComputeDescriptors(const std::vector<Mat>& gaussian_pyramid,
                               const std::vector<SiftKeyPoint>& keypoints, Mat& descriptors,
                               int num_split_in_octave, int first_octave) {
  // Parallelize with respect to keypoints.
  parallel_for_(Range(0, static_cast<int>(keypoints.size())),
                ComputeDescriptorComputer(gaussian_pyramid, keypoints, descriptors,
                                          num_split_in_octave, first_octave));
}

SIFT::SIFT(int num_features, int num_split_in_octave, double contrast_threshold,
           double edge_threshold, double _sigma)
    : num_features_(num_features),
      num_split_in_octave_(num_split_in_octave),
      num_gaussian_in_octave_(num_split_in_octave + 3),
      num_diff_of_gaussian_in_octave_(num_split_in_octave + 2),
      contrast_threshold_(contrast_threshold),
      edge_threshold_(edge_threshold),
      sigma(_sigma) {}

void SIFT::Detect(cv::Mat& image, std::vector<SiftKeyPoint>& keypoints) {
  int first_octave_idx = -1;

  // Initialize Image.
  bool double_image_size = true;
  Mat base_image = CreateInitialImage(image, double_image_size, (float)sigma);

  // Compute number of octave that is solely computed from image size.
  double log2_num = std::log((double)std::min(base_image.cols, base_image.rows)) / std::log(2.);
  int num_octaves = cvRound(log2_num - 2) + 1;

  // Build Gaussian Pyramid.
  std::vector<Mat> gaussian_pyr;
  BuildGaussianPyramid(base_image, gaussian_pyr, num_octaves);

  // Build Difference of Gaussian Pyramid.
  std::vector<Mat> diff_of_gaussian_pyr;
  BuildDifferenceOfGaussianPyramid(gaussian_pyr, diff_of_gaussian_pyr);

  // Find Scale Space Extrema in Difference of Gaussian Pyramid.
  FindScaleSpaceExtrema(gaussian_pyr, diff_of_gaussian_pyr, keypoints);

  // Remove Duplicated Key Point.
  RemoveDuplicateSorted(keypoints);

  // If maximum number supeciried.
  if (num_features_ > 0) {
    RetainBest(keypoints, num_features_);
  }

  // Update keypoints.
  for (size_t i = 0; i < keypoints.size(); i++) {
    SiftKeyPoint& kpt = keypoints[i];
    float scale = std::pow(2.0, first_octave_idx);
    kpt.octave = kpt.octave + first_octave_idx;
    kpt.pt *= scale;
    kpt.size *= scale;
  }
}

void SIFT::Compute(cv::Mat& image, std::vector<SiftKeyPoint>& keypoints, cv::Mat& descriptors) {
  // Initialization based on keypoints information.
  int first_octave = 0;
  int total_octaves;
  Mat base;
  {
    // Structure information extracted from keypoints.
    int last_octave = INT_MIN;
    for (auto kpt : keypoints) {
      first_octave = std::min(first_octave, kpt.octave);
      last_octave = std::max(last_octave, kpt.octave);
    }
    total_octaves = last_octave - first_octave + 1;

    // Create initial image from the given.
    base = CreateInitialImage(image, first_octave < 0, (float)sigma);
  }

  // Build Gaussian Pyramid.
  std::vector<Mat> gaussian_pyramid;
  { BuildGaussianPyramid(base, gaussian_pyramid, total_octaves); }

  // Compute descriptor for each keypoint.
  {
    descriptors.create((int)keypoints.size(), DescriptorSize(), CV_32F);
    ComputeDescriptors(gaussian_pyramid, keypoints, descriptors, num_split_in_octave_,
                       first_octave);
  }
}

void SIFT::BuildGaussianPyramid(const Mat& base_image, std::vector<Mat>& pyramid, int nOctaves,
                                bool debug_display) const {
  // Compute sigmas for gaussian images in octave.
  // precompute Gaussian sigmas using the following formula:
  //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
  std::vector<double> sigmas_for_gaussian(num_gaussian_in_octave_);
  {
    sigmas_for_gaussian[0] = sigma;
    double k = std::pow(2., 1. / num_split_in_octave_);
    // Loop for gaussian images in octave.
    for (int i = 1; i < num_gaussian_in_octave_; i++) {
      double sigma_previous = std::pow(k, (double)(i - 1)) * sigma;
      double sigma_total = sigma_previous * k;
      sigmas_for_gaussian[i] =
          std::sqrt(sigma_total * sigma_total - sigma_previous * sigma_previous);
    }
  }

  // Building Pyramid. Loop for all octave and all layer inside them.
  {
    pyramid.resize(nOctaves * (num_gaussian_in_octave_));
    for (int oct_idx = 0; oct_idx < nOctaves; oct_idx++) {
      for (int layer_idx = 0; layer_idx < num_gaussian_in_octave_; layer_idx++) {
        Mat& dst = pyramid[oct_idx * (num_gaussian_in_octave_) + layer_idx];
        if (oct_idx == 0 && layer_idx == 0) {
          dst = base_image;
        } else if (layer_idx == 0) {
          // Switch to New Octave. New octave is halved image from end of previous octave.
          const Mat& src =
              pyramid[(oct_idx - 1) * (num_gaussian_in_octave_) + num_split_in_octave_];
          resize(src, dst, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_NEAREST);
        } else {
          const Mat& src = pyramid[oct_idx * (num_gaussian_in_octave_) + layer_idx - 1];
          GaussianBlur(src, dst, Size(), sigmas_for_gaussian[layer_idx],
                       sigmas_for_gaussian[layer_idx]);
        }
        if (debug_display) {
          DisplayImage(dst, "Pyramid", 0);
        }
      }
    }
  }
}

void SIFT::BuildDifferenceOfGaussianPyramid(const std::vector<Mat>& gaussian_pyramid,
                                            std::vector<Mat>& diff_of_gaussian_pyramid,
                                            bool debug_display) const {
  // Num of images in octave.
  int num_octaves = (int)gaussian_pyramid.size() / (num_gaussian_in_octave_);

  // Number of difference of gaussian image.
  diff_of_gaussian_pyramid.resize(num_octaves * num_diff_of_gaussian_in_octave_);

  //　Pararellization for computing keypoints.
  parallel_for_(Range(0, num_octaves * num_diff_of_gaussian_in_octave_),
                BuildDiffOfGaussianPyramid(num_gaussian_in_octave_, num_diff_of_gaussian_in_octave_,
                                           gaussian_pyramid, diff_of_gaussian_pyramid));

  // Display intermediate results.
  if (debug_display) {
    for (auto dog_img : diff_of_gaussian_pyramid) {
      DisplayImage(dog_img, "Difference of Gaussian", 0);
    }
  }
}

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void SIFT::FindScaleSpaceExtrema(const std::vector<Mat>& gaussian_pyramid,
                                 const std::vector<Mat>& diff_of_gaussian_pyramid,
                                 std::vector<SiftKeyPoint>& keypoints) const {
  const int num_octave = (int)gaussian_pyramid.size() / (num_gaussian_in_octave_);
  const int threshold =
      cvFloor(0.5 * contrast_threshold_ / num_split_in_octave_ * 255 * SIFT_FIXPT_SCALE);

  keypoints.clear();
  TLSData<std::vector<SiftKeyPoint> > tls_kpts_struct;

  // Loop for octave.
  for (int oct_idx = 0; oct_idx < num_octave; oct_idx++) {
    // Loop for split of one octave.
    for (int split_idx = 1; split_idx <= num_split_in_octave_; split_idx++) {
      const int dog_idx = oct_idx * (num_diff_of_gaussian_in_octave_) + split_idx;

      const Mat& diff_of_gaussian = diff_of_gaussian_pyramid[dog_idx];
      const int step = (int)diff_of_gaussian.step1();
      const int rows = diff_of_gaussian.rows, cols = diff_of_gaussian.cols;

      // Pararellization for each row.
      parallel_for_(Range(SIFT_IMG_BORDER, rows - SIFT_IMG_BORDER),
                    FindScaleSpaceExtremaComputer(oct_idx, split_idx, threshold, dog_idx, step,
                                                  cols, num_split_in_octave_, contrast_threshold_,
                                                  edge_threshold_, sigma, gaussian_pyramid,
                                                  diff_of_gaussian_pyramid, tls_kpts_struct));
    }
  }

  std::vector<std::vector<SiftKeyPoint>*> kpt_vecs;
  tls_kpts_struct.gather(kpt_vecs);
  for (size_t i = 0; i < kpt_vecs.size(); ++i) {
    keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
  }
}

int SIFT::DescriptorSize() const {
  return SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
}

int SIFT::DescriptorType() const { return CV_32F; }

int SIFT::DefaultNorm() const { return NORM_L2; }

}  // namespace cv_copy