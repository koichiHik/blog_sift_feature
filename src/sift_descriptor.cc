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

static void ComputeGradientForPrePatch(const float angle, const int pre_patch_size,
                                       const float pre_patch_radius, const int final_patch_width,
                                       const float radius, const Point2i pt, const Mat img,
                                       std::vector<float>& RBin, std::vector<float>& CBin,
                                       std::vector<float>& Ori, std::vector<float>& Mag,
                                       std::vector<float>& W) {
  {
    float exp_scale = -1.f / (final_patch_width * final_patch_width * 0.5f);

    // For coordinate transform.
    float cos_t = cosf(angle * (float)(CV_PI / 180));
    float sin_t = sinf(angle * (float)(CV_PI / 180));

    // Division by pre_patch_radius for length normalizing.
    cos_t /= pre_patch_radius;
    sin_t /= pre_patch_radius;

    // Prepare temporary buffer.
    std::vector<float> X(pre_patch_size, 0.0f), Y(pre_patch_size, 0.0f);

    // Compute gradients and weighted gradients of pixels around keypoint.
    int pix_cnt = 0;
    for (int dy = -radius; dy <= radius; dy++) {
      for (int dx = -radius; dx <= radius; dx++) {
        // Rotate for keypoint orientation.
        float dx_rot = (dx * cos_t - dy * sin_t);
        float dy_rot = (dx * sin_t + dy * cos_t);

        // Calculate sample's histogram array coords rotated relative to ori.
        // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
        // r_rot = 1.5) have full weight placed in row 1 after interpolation.

        // Compute location in final patch coordinate.
        float final_patch_col = dx_rot + final_patch_width / 2 - 0.5f;
        float final_patch_row = dy_rot + final_patch_width / 2 - 0.5f;
        int row = pt.y + dy, col = pt.x + dx;

        if (-1 < final_patch_row && final_patch_row < final_patch_width && -1 < final_patch_col &&
            final_patch_col < final_patch_width && 0 < row && row < img.rows - 1 && 0 < col &&
            col < img.cols - 1) {
          X[pix_cnt] = (float)(img.at<sift_wt>(row, col + 1) - img.at<sift_wt>(row, col - 1));
          Y[pix_cnt] = (float)(img.at<sift_wt>(row - 1, col) - img.at<sift_wt>(row + 1, col));
          RBin[pix_cnt] = final_patch_row;
          CBin[pix_cnt] = final_patch_col;
          W[pix_cnt] = (dx_rot * dx_rot + dy_rot * dy_rot) * exp_scale;
          pix_cnt++;
        }
      }
    }

    X.resize(pix_cnt);
    Y.resize(pix_cnt);
    Ori.resize(pix_cnt);
    Mag.resize(pix_cnt);
    W.resize(pix_cnt);

    cv::hal::fastAtan2(Y.data(), X.data(), Ori.data(), pix_cnt, true);
    cv::hal::magnitude32f(X.data(), Y.data(), Mag.data(), pix_cnt);
    cv::hal::exp32f(W.data(), W.data(), pix_cnt);
  }
}

static void UpdateHistogramBasedOnTriLinearInterpolation(
    const std::vector<float>& Mag, const std::vector<float>& W, const std::vector<float>& RBin,
    const std::vector<float>& CBin, const std::vector<float>& Ori, const float angle,
    const int final_ori_hist_bins, const int final_patch_width, std::vector<float>& hist) {
  float bin_resolution = final_ori_hist_bins / 360.f;

  for (int pix_idx = 0; pix_idx < Ori.size(); pix_idx++) {
    int row = cvFloor(RBin[pix_idx]);
    int col = cvFloor(CBin[pix_idx]);
    int ori = cvFloor((Ori[pix_idx] - angle) * bin_resolution);

    float d_row = RBin[pix_idx] - row;
    float d_col = CBin[pix_idx] - col;
    float d_ori = (Ori[pix_idx] - angle) * bin_resolution - ori;

    // Angle normalization.
    ori = (ori + final_ori_hist_bins) % final_ori_hist_bins;

    // Tri-linear interpolation before histogram update.
    float mag = Mag[pix_idx] * W[pix_idx];
    float v_r1 = mag * d_row;
    float v_r0 = mag - v_r1;
    float v_rc11 = v_r1 * d_col;
    float v_rc10 = v_r1 - v_rc11;
    float v_rc01 = v_r0 * d_col;
    float v_rc00 = v_r0 - v_rc01;
    float v_rco111 = v_rc11 * d_ori;
    float v_rco110 = v_rc11 - v_rco111;
    float v_rco101 = v_rc10 * d_ori;
    float v_rco100 = v_rc10 - v_rco101;
    float v_rco011 = v_rc01 * d_ori;
    float v_rco010 = v_rc01 - v_rco011;
    float v_rco001 = v_rc00 * d_ori;
    float v_rco000 = v_rc00 - v_rco001;

    // Histogram update.
    int idx = ((row + 1) * (final_patch_width + 2) + col + 1) * (final_ori_hist_bins + 2) + ori;

    // This buffer has 3 dimensional structure. (X x Y x Ori)
    // (row, col, ori)
    hist[idx] += v_rco000;
    // (row, col, ori + 1)
    hist[idx + 1] += v_rco001;
    // (row, col + 1, ori)
    hist[idx + (final_ori_hist_bins + 2)] += v_rco010;
    // (row, col + 1, ori + 1)
    hist[idx + (final_ori_hist_bins + 2) + 1] += v_rco011;
    // (row + 1, col, ori)
    hist[idx + (final_patch_width + 2) * (final_ori_hist_bins + 2)] += v_rco100;
    // (row + 1, col, ori + 1)
    hist[idx + (final_patch_width + 2) * (final_ori_hist_bins + 2) + 1] += v_rco101;
    // (row + 1, col + 1, ori)
    hist[idx + ((final_patch_width + 2) + 1) * (final_ori_hist_bins + 2)] += v_rco110;
    // (row + 1, col + 1, ori + 1)
    hist[idx + ((final_patch_width + 2) + 1) * (final_ori_hist_bins + 2) + 1] += v_rco111;
  }
}

static void RefineHistogram(const int final_patch_width, const int final_ori_hist_bins,
                            std::vector<float>& hist, float* dst) {
  // Finalize histogram, since the orientation histograms are circular.
  for (int row = 0; row < final_patch_width; row++) {
    for (int col = 0; col < final_patch_width; col++) {
      int idx = ((row + 1) * (final_patch_width + 2) + (col + 1)) * (final_ori_hist_bins + 2);

      hist[idx] += hist[idx + final_ori_hist_bins];
      hist[idx + 1] += hist[idx + final_ori_hist_bins + 1];

      // Loop for orientation histogram of (row, col)
      for (int ori = 0; ori < final_ori_hist_bins; ori++) {
        dst[(row * final_patch_width + col) * final_ori_hist_bins + ori] = hist[idx + ori];
      }
    }
  }

  // Refine histogram.
  {
    // Apply hysterisis thresholding and scale the result for easy converion to byte array.
    int size = final_patch_width * final_patch_width * final_ori_hist_bins;

    // Find thresh.
    float thresh;
    {
      float total_norm = 0;
      for (int k = 0; k < size; k++) {
        total_norm += dst[k] * dst[k];
      }
      thresh = std::sqrt(total_norm) * Const::SIFT_DESCR_MAG_THR;
    }

    // Normalize the result.
    {
      float total_norm = 0;
      for (int k = 0; k < size; k++) {
        float val = std::min(dst[k], thresh);
        dst[k] = val;
        total_norm += val * val;
      }
      total_norm = Const::SIFT_INT_DESCR_FCTR / std::max(std::sqrt(total_norm), FLT_EPSILON);

      for (int k = 0; k < size; k++) {
        dst[k] = saturate_cast<uchar>(dst[k] * total_norm);
      }
    }
  }
}

// Compute sift descriptor.
static void ComputeSIFTDescriptor(const Mat& img, Point2f ptf, float angle, float scl,
                                  int final_patch_width, int final_ori_hist_bins, float* dst) {
  // 1. Patch size calculation.
  int radius, pre_patch_size, final_hist_size;
  float pre_patch_radius;
  {
    pre_patch_radius = Const::SIFT_DESCR_SCL_FCTR * scl;
    // Compute radius.
    float sqrt2 = 1.4142135623730951f;
    radius = cvRound(pre_patch_radius * sqrt2 * (final_patch_width + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius =
        std::min(radius, (int)sqrt(((double)img.cols) * img.cols + ((double)img.rows) * img.rows));
    pre_patch_size = (radius * 2 + 1) * (radius * 2 + 1);
    final_hist_size = (final_patch_width + 2) * (final_patch_width + 2) * (final_ori_hist_bins + 2);
  }

  // 2. Prepare buffer.
  std::vector<float> Mag(pre_patch_size, 0.0f), Ori(pre_patch_size, 0.0f), W(pre_patch_size, 0.0f),
      RBin(pre_patch_size, 0.0f), CBin(pre_patch_size, 0.0f), hist(final_hist_size, 0.0f);

  // 3. Computation for pre-patch element.
  ComputeGradientForPrePatch(angle, pre_patch_size, pre_patch_radius, final_patch_width, radius,
                             Point2i(cvRound(ptf.x), cvRound(ptf.y)), img, RBin, CBin, Ori, Mag, W);

  // 4. Histogram update based on tri-linear interpolation.
  UpdateHistogramBasedOnTriLinearInterpolation(Mag, W, RBin, CBin, Ori, angle, final_ori_hist_bins,
                                               final_patch_width, hist);

  // 5. Refine histogram.
  RefineHistogram(final_patch_width, final_ori_hist_bins, hist, dst);
}

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

    static const int desc_patch_width = Const::SIFT_DESCR_WIDTH;
    static const int desc_hist_bins = Const::SIFT_DESCR_HIST_BINS;

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

}  // namespace

namespace cv_copy {

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
    base = CreateInitialImage(image, first_octave < 0, (float)sigma_);
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

}  // namespace cv_copy