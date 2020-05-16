/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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

#ifndef __SIFT__HPP__
#define __SIFT__HPP__

// STL
#include <iostream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv_copy {

using namespace cv;

typedef float sift_wt;

class Const {
 public:
  // default width of descriptor histogram array
  static inline const int SIFT_DESCR_WIDTH = 4;

  // default number of bins per histogram in descriptor array
  static inline const int SIFT_DESCR_HIST_BINS = 8;

  // assumed gaussian blur for input image
  static inline const float SIFT_INIT_SIGMA = 0.5f;

  // width of border in which to ignore keypoints
  static inline const int SIFT_IMG_BORDER = 5;

  // maximum steps of keypoint interpolation before failure
  static inline const int SIFT_MAX_INTERP_STEPS = 5;

  // default number of bins in histogram for orientation assignment
  static inline const int SIFT_ORI_HIST_BINS = 36;

  // determines gaussian sigma for orientation assignment
  static inline const float SIFT_ORI_SIG_FCTR = 1.5f;

  // determines the radius of the region used in orientation assignment
  static inline const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

  // orientation magnitude relative to max that results in new feature
  static inline const float SIFT_ORI_PEAK_RATIO = 0.8f;

  // determines the size of a single descriptor orientation histogram
  static inline const float SIFT_DESCR_SCL_FCTR = 3.f;

  // threshold on magnitude of elements of descriptor vector
  static inline const float SIFT_DESCR_MAG_THR = 0.2f;

  // factor used to convert floating-point descriptor to unsigned char
  static inline const float SIFT_INT_DESCR_FCTR = 512.f;

  // intermediate type used for DoG pyramids
  static inline const int SIFT_FIXPT_SCALE = 1;
};

class SiftKeyPoint : public KeyPoint {
 public:
  SiftKeyPoint() : layer(0){};
  int layer;
};

class SIFT {
 public:
  explicit SIFT(int num_features = 0, int num_split_in_octave = 3, double contrast_threshold = 0.04,
                double edge_threshold = 10, double sigma = 1.6)
      : num_features_(num_features),
        num_split_in_octave_(num_split_in_octave),
        num_gaussian_in_octave_(num_split_in_octave + 3),
        num_diff_of_gaussian_in_octave_(num_split_in_octave + 2),
        contrast_threshold_(contrast_threshold),
        edge_threshold_(edge_threshold),
        sigma_(sigma){};

  void Detect(cv::Mat& image, std::vector<SiftKeyPoint>& keypoints);

  void Compute(cv::Mat& image, std::vector<SiftKeyPoint>& keypoints, cv::Mat& descriptors);

 protected:
  void BuildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int num_octaves) const;

  void BuildDifferenceOfGaussianPyramid(const std::vector<Mat>& gpyr,
                                        std::vector<Mat>& dogpyr) const;

  void FindScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                             std::vector<SiftKeyPoint>& keypoints) const;

  int DescriptorSize() const {
    return Const::SIFT_DESCR_WIDTH * Const::SIFT_DESCR_WIDTH * Const::SIFT_DESCR_HIST_BINS;
  };
  int DescriptorType() const { return CV_32F; };
  int DefaultNorm() const { return NORM_L2; };

 protected:
  int num_features_;
  int num_split_in_octave_;
  int num_gaussian_in_octave_;
  int num_diff_of_gaussian_in_octave_;
  double contrast_threshold_;
  double edge_threshold_;
  double sigma_;
};

}  // namespace cv_copy

#endif