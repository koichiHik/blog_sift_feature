
#ifndef __OPENCV_COPY_ORG_XFEATURES2D_FEATURES_2D_HPP__
#define __OPENCV_COPY_ORG_XFEATURES2D_FEATURES_2D_HPP__

#include "opencv2/features2d.hpp"

namespace cv_copy {
namespace xfeatures2d {

using namespace cv;

class SIFT : public Feature2D {
 public:
  static Ptr<SIFT> create(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
                          double edgeThreshold = 10, double sigma = 1.6);
};

}  // namespace xfeatures2d
}  // namespace cv_copy

#endif