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
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Glog
#include <glog/logging.h>

// Original
#include "sift.hpp"

using namespace cv;
using namespace cv_copy;

namespace {

static Mat CreateInitialImage(const Mat& img, bool doubleImageSize, float sigma) {
  Mat gray, gray_fpt;
  if (img.channels() == 3 || img.channels() == 4) {
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray_fpt, DataType<sift_wt>::type, Const::SIFT_FIXPT_SCALE, 0);
  } else {
    img.convertTo(gray_fpt, DataType<sift_wt>::type, Const::SIFT_FIXPT_SCALE, 0);
  }

  float sig_diff;

  if (doubleImageSize) {
    sig_diff =
        sqrtf(std::max(sigma * sigma - Const::SIFT_INIT_SIGMA * Const::SIFT_INIT_SIGMA * 4, 0.01f));
    Mat dbl;
    resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR);
    GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
    return dbl;
  } else {
    sig_diff =
        sqrtf(std::max(sigma * sigma - Const::SIFT_INIT_SIGMA * Const::SIFT_INIT_SIGMA, 0.01f));
    GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
    return gray_fpt;
  }
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

static void ComputeValuesForPatch(const Mat& img, const Point2i& pt, const int radius,
                                  const float sigma, std::vector<float>& Ori,
                                  std::vector<float>& Mag, std::vector<float>& W) {
  int patch_size = (radius * 2 + 1) * (radius * 2 + 1);

  // Loop for square that encircles keypoint.
  // Compute gradient and magnitude.
  {
    int kpt_cnt = 0;
    std::vector<float> X(patch_size), Y(patch_size);
    float expf_scale = -1.f / (2.f * sigma * sigma);

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
        X[kpt_cnt] = (float)(img.at<sift_wt>(y, x + 1) - img.at<sift_wt>(y, x - 1));
        Y[kpt_cnt] = (float)(img.at<sift_wt>(y - 1, x) - img.at<sift_wt>(y + 1, x));
        W[kpt_cnt] = (dx * dx + dy * dy) * expf_scale;
        kpt_cnt++;
      }
    }

    W.resize(kpt_cnt);
    X.resize(kpt_cnt);
    Y.resize(kpt_cnt);
    W.resize(kpt_cnt);
    Ori.resize(kpt_cnt);
    Mag.resize(kpt_cnt);

    // Compute necessary buffer.
    cv::hal::exp32f(W.data(), W.data(), kpt_cnt);
    cv::hal::fastAtan2(Y.data(), X.data(), Ori.data(), kpt_cnt, true);
    cv::hal::magnitude32f(X.data(), Y.data(), Mag.data(), kpt_cnt);
  }
}

static float ComputeHistogram(const int num_hist_bins, const int patch_size,
                              const std::vector<float>& Ori, const std::vector<float>& Mag,
                              const std::vector<float>& W, float* hist) {
  // Orientation Histogram Computation.
  float max_val;
  {
    std::vector<float> temphist(patch_size * 2, 0.0f);
    // Compute gradient values, orientations and the weights over the pixel neighborhood.
    for (int k = 0; k < Ori.size(); k++) {
      int bin = cvRound((num_hist_bins / 360.f) * Ori[k]);
      bin = (num_hist_bins + bin) % num_hist_bins;
      temphist[bin] += W[k] * Mag[k];
    }

    for (int bins = 0; bins < num_hist_bins; bins++) {
      // Since histogram is circular.
      int right_2_idx = (num_hist_bins + bins - 2) % num_hist_bins;
      int right_1_idx = (num_hist_bins + bins - 1) % num_hist_bins;
      int center_idx = bins;
      int left_1_idx = (num_hist_bins + bins + 1) % num_hist_bins;
      int left_2_idx = (num_hist_bins + bins + 2) % num_hist_bins;

      hist[bins] = (temphist[right_2_idx] + temphist[left_2_idx]) * (1.f / 16.f) +
                   (temphist[right_1_idx] + temphist[left_1_idx]) * (4.f / 16.f) +
                   temphist[center_idx] * (6.f / 16.f);
    }

    max_val = hist[0];
    for (int bins = 1; bins < num_hist_bins; bins++) {
      max_val = std::max(max_val, hist[bins]);
    }
  }

  return max_val;
}

// Computes a gradient orientation histogram at a specified pixel
static float ComputeOrientationHistogram(const Mat& img, const Point2i& pt, int radius, float sigma,
                                         float* hist, int num_hist_bins) {
  int patch_size = (radius * 2 + 1) * (radius * 2 + 1);

  std::vector<float> Ori(patch_size), Mag(patch_size), W(patch_size);
  ComputeValuesForPatch(img, pt, radius, sigma, Ori, Mag, W);

  // Compute orientation histogram.
  float max_val = ComputeHistogram(num_hist_bins, patch_size, Ori, Mag, W, hist);

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

static void CompensateInitialScalingToOriginalImage(std::vector<SiftKeyPoint>& keypoints,
                                                    int first_octave_idx) {
  // Adjust keypoint scale wrt first octave setting we apply.
  for (size_t i = 0; i < keypoints.size(); i++) {
    SiftKeyPoint& kpt = keypoints[i];
    float scale = std::pow(2.0, first_octave_idx);
    kpt.octave = kpt.octave + first_octave_idx;
    kpt.pt *= scale;
    kpt.size *= scale;
  }
}

static bool RefineKeyPointLocationIntoSubPixel(
    const int num_split_in_octave, const int octv, const float contrast_threshold,
    const float edge_threshold, const float sigma, const float deriv_scale,
    const float second_deriv_scale, const float cross_deriv_scale,
    const std::vector<Mat>& diff_of_gaussian_pyr, Vec3i& X, Vec3f& dX) {
  // Copy into modifiable buffer.

  for (int i = 0; i <= Const::SIFT_MAX_INTERP_STEPS; i++) {
    if (i >= Const::SIFT_MAX_INTERP_STEPS) {
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

    if (X[2] < 1 || X[2] > num_split_in_octave || X[0] < Const::SIFT_IMG_BORDER ||
        X[0] >= mid_dog.cols - Const::SIFT_IMG_BORDER || X[1] < Const::SIFT_IMG_BORDER ||
        X[1] >= mid_dog.rows - Const::SIFT_IMG_BORDER) {
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
  const float img_scale = 1.f / (255 * Const::SIFT_FIXPT_SCALE);
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

static void ComputeRepresentativeOrientationForKeyPoint(const int num_hist_bins,
                                                        const float orientation_max_val,
                                                        const float* hist, SiftKeyPoint kpt,
                                                        std::vector<SiftKeyPoint>* tls_kpts) {
  // Compute threshold above which the orientation will be adopted.
  float orientation_threshold = (float)(orientation_max_val * Const::SIFT_ORI_PEAK_RATIO);
  for (int cur_bin = 0; cur_bin < num_hist_bins; cur_bin++) {
    int left_bin = (num_hist_bins + cur_bin - 1) % num_hist_bins;
    int right_bin = (num_hist_bins + cur_bin + 1) % num_hist_bins;

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

static void ComputeOrientationForKeyPoint(const int oct_idx_, const int num_split_in_octave_,
                                          const int cand_row, const int cand_col, const int layer,
                                          const std::vector<Mat> gaussian_pyramid_,
                                          const int num_hist_bins, float* hist, SiftKeyPoint& kpt,
                                          std::vector<SiftKeyPoint>* tls_kpts) {
  // Compute orientation.
  float scl_octv = kpt.size * 0.5f / (1 << oct_idx_);

  float orientation_max_val = ComputeOrientationHistogram(
      gaussian_pyramid_[oct_idx_ * (num_split_in_octave_ + 3) + layer], Point2i(cand_col, cand_row),
      cvRound(Const::SIFT_ORI_RADIUS * scl_octv), Const::SIFT_ORI_SIG_FCTR * scl_octv, hist,
      num_hist_bins);

  ComputeRepresentativeOrientationForKeyPoint(num_hist_bins, orientation_max_val, hist, kpt,
                                              tls_kpts);
}

class FindScaleSpaceExtremaComputer : public ParallelLoopBody {
 public:
  FindScaleSpaceExtremaComputer(int oct_idx, int split_idx, int threshold, int dog_idx, int stride,
                                int cols, int num_split_in_octave, double contrast_threshold,
                                double edge_threshold, double sigma,
                                const std::vector<Mat>& gaussian_pyramid,
                                const std::vector<Mat>& diff_of_gaussian_pyramid,
                                TLSDataAccumulator<std::vector<SiftKeyPoint> >& _tls_kpts_struct)

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

    static const int num_hist_bins = Const::SIFT_ORI_HIST_BINS;
    float hist[num_hist_bins];
    SiftKeyPoint kpt;
    // Outer loop is for row.
    for (int row = begin; row < end; row++) {
      const sift_wt* prevptr = prev.ptr<sift_wt>(row);
      const sift_wt* currptr = img.ptr<sift_wt>(row);
      const sift_wt* nextptr = next.ptr<sift_wt>(row);

      // Inner loop is for col.
      for (int col = Const::SIFT_IMG_BORDER; col < cols_ - Const::SIFT_IMG_BORDER; col++) {
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

}  // namespace

namespace cv_copy {

void SIFT::Detect(cv::Mat& image, std::vector<SiftKeyPoint>& keypoints) {
  int first_octave_idx = -1;

  // 1. Initialize image.
  bool double_image_size = true;
  Mat base_image = CreateInitialImage(image, double_image_size, (float)sigma_);

  // 2. Compute number of octave that is solely computed from image size.
  double log2_num = std::log((double)std::min(base_image.cols, base_image.rows)) / std::log(2.);
  int num_octaves = cvRound(log2_num - 2) + 1;

  // 3. Build Gaussian pyramid.
  std::vector<Mat> gaussian_pyr;
  BuildGaussianPyramid(base_image, gaussian_pyr, num_octaves);

  // 4. Build difference of Gaussian pyramid.
  std::vector<Mat> diff_of_gaussian_pyr;
  BuildDifferenceOfGaussianPyramid(gaussian_pyr, diff_of_gaussian_pyr);

  // 5. Find scale space extrema in difference of Gaussian pyramid.
  FindScaleSpaceExtrema(gaussian_pyr, diff_of_gaussian_pyr, keypoints);

  // 6. Remove duplicated key point.
  RemoveDuplicateSorted(keypoints);

  // 7. If maximum number supeciried.
  if (num_features_ > 0) {
    RetainBest(keypoints, num_features_);
  }

  // 8. Compensating doubling of iamge at the beginning.
  CompensateInitialScalingToOriginalImage(keypoints, first_octave_idx);
}

void SIFT::BuildGaussianPyramid(const Mat& base_image, std::vector<Mat>& pyramid,
                                int num_octaves) const {
  // Compute sigmas for gaussian images in octave.
  // precompute Gaussian sigmas using the following formula:
  //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
  std::vector<double> sigmas_for_gaussian(num_gaussian_in_octave_);
  {
    sigmas_for_gaussian[0] = sigma_;
    double k = std::pow(2., 1. / num_split_in_octave_);
    // Loop for gaussian images in octave.
    for (int i = 1; i < num_gaussian_in_octave_; i++) {
      double sigma_previous = std::pow(k, (double)(i - 1)) * sigma_;
      double sigma_total = sigma_previous * k;
      sigmas_for_gaussian[i] =
          std::sqrt(sigma_total * sigma_total - sigma_previous * sigma_previous);
    }
  }

  // Building Pyramid. Loop for all octave and all layer inside them.
  {
    pyramid.resize(num_octaves * (num_gaussian_in_octave_));
    for (int oct_idx = 0; oct_idx < num_octaves; oct_idx++) {
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
      }
    }
  }
}

void SIFT::BuildDifferenceOfGaussianPyramid(const std::vector<Mat>& gaussian_pyramid,
                                            std::vector<Mat>& diff_of_gaussian_pyramid) const {
  // Num of images in octave.
  int num_octaves = (int)gaussian_pyramid.size() / (num_gaussian_in_octave_);

  // Number of difference of gaussian image.
  diff_of_gaussian_pyramid.resize(num_octaves * num_diff_of_gaussian_in_octave_);

  //　Pararellization for computing keypoints.
  parallel_for_(Range(0, num_octaves * num_diff_of_gaussian_in_octave_),
                BuildDiffOfGaussianPyramid(num_gaussian_in_octave_, num_diff_of_gaussian_in_octave_,
                                           gaussian_pyramid, diff_of_gaussian_pyramid));
}

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void SIFT::FindScaleSpaceExtrema(const std::vector<Mat>& gaussian_pyramid,
                                 const std::vector<Mat>& diff_of_gaussian_pyramid,
                                 std::vector<SiftKeyPoint>& keypoints) const {
  const int num_octave = (int)gaussian_pyramid.size() / (num_gaussian_in_octave_);
  const int threshold =
      cvFloor(0.5 * contrast_threshold_ / num_split_in_octave_ * 255 * Const::SIFT_FIXPT_SCALE);

  keypoints.clear();
  //TLSData<std::vector<SiftKeyPoint> > tls_kpts_struct;
  TLSDataAccumulator<std::vector<SiftKeyPoint> > tls_kpts_struct;


  // Loop for octave.
  for (int oct_idx = 0; oct_idx < num_octave; oct_idx++) {
    // Loop for split of one octave.
    for (int split_idx = 1; split_idx <= num_split_in_octave_; split_idx++) {
      const int dog_idx = oct_idx * (num_diff_of_gaussian_in_octave_) + split_idx;

      const Mat& diff_of_gaussian = diff_of_gaussian_pyramid[dog_idx];
      const int step = (int)diff_of_gaussian.step1();
      const int rows = diff_of_gaussian.rows, cols = diff_of_gaussian.cols;

      // Pararellization for each row.
      parallel_for_(Range(Const::SIFT_IMG_BORDER, rows - Const::SIFT_IMG_BORDER),
                    FindScaleSpaceExtremaComputer(oct_idx, split_idx, threshold, dog_idx, step,
                                                  cols, num_split_in_octave_, contrast_threshold_,
                                                  edge_threshold_, sigma_, gaussian_pyramid,
                                                  diff_of_gaussian_pyramid, tls_kpts_struct));
    }
  }

  std::vector<std::vector<SiftKeyPoint>*> kpt_vecs;
  tls_kpts_struct.gather(kpt_vecs);
  for (size_t i = 0; i < kpt_vecs.size(); ++i) {
    keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
  }
}

}  // namespace cv_copy