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
  explicit SIFT(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
                double edgeThreshold = 10, double sigma = 1.6);

  void detect(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask = noArray());

  void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors);

  /** Detects keypoints and computes the descriptors */
  void detectAndCompute(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                        OutputArray descriptors, bool useProvidedKeypoints = false);

 protected:
  void buildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves) const;

  void buildDoGPyramid(const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr) const;

  void findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                             std::vector<KeyPoint>& keypoints) const;

  int descriptorSize() const;
  int descriptorType() const;
  int defaultNorm() const;

 protected:
  int nfeatures;
  int nOctaveLayers;
  double contrastThreshold;
  double edgeThreshold;
  double sigma;
};

}  // namespace cv_copy
