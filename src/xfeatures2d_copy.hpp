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

// STL
#include <iostream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv_copy {

using namespace cv;

class SIFT {
 public:
  explicit SIFT(int num_features = 0, int num_split_in_octave = 3, double contrast_threshold = 0.04,
                double edge_threshold = 10, double sigma = 1.6);

  void Detect(cv::Mat& image, std::vector<KeyPoint>& keypoints);

  void Compute(cv::Mat& image, std::vector<KeyPoint>& keypoints, cv::Mat& descriptors);

 protected:
  void BuildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves,
                            bool debug_display = false) const;

  void BuildDifferenceOfGaussianPyramid(const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr,
                                        bool debug_display = false) const;

  void FindScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                             std::vector<KeyPoint>& keypoints) const;

  int DescriptorSize() const;
  int DescriptorType() const;
  int DefaultNorm() const;

 protected:
  int num_features_;
  int num_split_in_octave_;
  int num_gaussian_in_octave_;
  int num_diff_of_gaussian_in_octave_;
  double contrast_threshold_;
  double edge_threshold_;
  double sigma;
};

}  // namespace cv_copy
