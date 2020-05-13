

#include <iomanip>
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>

// Eigen
#include <eigen3/Eigen/Core>

// GFLAGS
#include <gflags/gflags.h>

// GLOG
#include <glog/logging.h>

// Original
#include "xfeatures2d_copy.hpp"
#include "xfeatures2d_copy_org.hpp"

using namespace cv_copy;

DEFINE_string(image_path, "", "Path to the image.");

template <typename T>
void DrawKeyPointsAsCircles(const cv::Mat& img, const std::vector<T>& keypoints, cv::Mat& result) {
  img.copyTo(result);
  for (auto pnt : keypoints) {
    cv::Point2i location(pnt.pt.x + 0.5, pnt.pt.y + 0.5);
    cv::circle(result, location, static_cast<int>(pnt.size + 0.5f), cv::Scalar(0, 0, 255));
  }
}

bool KeyPointsAreSame(std::vector<cv::KeyPoint>& org_keypoints,
                      std::vector<SiftKeyPoint>& mod_keypoints) {
  if (org_keypoints.size() != mod_keypoints.size()) {
    return false;
  }

  for (int idx = 0; idx < org_keypoints.size(); idx++) {
    const cv::KeyPoint& org_pnt = org_keypoints[idx];
    const SiftKeyPoint& mod_pnt = mod_keypoints[idx];

    if (org_pnt.pt.x != mod_pnt.pt.x || org_pnt.pt.y != mod_pnt.pt.y) {
      return false;
    }
    if (org_pnt.response != mod_pnt.response) {
      return false;
    }
    if (org_pnt.size != mod_pnt.size) {
      return false;
    }
    if (org_pnt.angle != mod_pnt.angle) {
      return false;
    }
    // Only last 16bit of octave is used.
    int dec_org_octave = (int)(org_pnt.octave & 255);
    if (dec_org_octave >= 128) {
      dec_org_octave = (dec_org_octave | -128);
    }
    if (dec_org_octave != mod_pnt.octave) {
      return false;
    }
    int dec_org_layer = (int)(org_pnt.octave >> 8) & 255;
    if (dec_org_layer != mod_pnt.layer) {
      return false;
    }
  }

  return true;
}

bool DescriptorsAreSame(cv::Mat cv_descriptors, cv::Mat mod_descriptors) {
  cv::Mat comparison;
  cv::bitwise_xor(cv_descriptors, mod_descriptors, comparison);
  return cv::countNonZero(comparison) == 0;
}

void TestSiftFeatures(cv::Mat& img) {
  // cv::setNumThreads(0);

  std::vector<SiftKeyPoint> keypoints;
  std::vector<cv::KeyPoint> cv_keypoints;
  {
    cv_copy::SIFT sift_detector;
    sift_detector.Detect(img, keypoints);
    cv::Ptr<cv_copy::xfeatures2d::SIFT> cv_sift_detector = cv_copy::xfeatures2d::SIFT::create();
    cv_sift_detector->detect(img, cv_keypoints);
    if (!KeyPointsAreSame(cv_keypoints, keypoints)) {
      LOG(INFO) << "Two keypoints are different.";
    }
  }

  cv::Mat cv_descriptors, descriptors;
  {
    cv_copy::SIFT sift_detector;
    sift_detector.Compute(img, keypoints, descriptors);
    cv::Ptr<cv_copy::xfeatures2d::SIFT> cv_sift_detector = cv_copy::xfeatures2d::SIFT::create();
    cv_sift_detector->compute(img, cv_keypoints, cv_descriptors);
    if (!DescriptorsAreSame(descriptors, cv_descriptors)) {
      LOG(INFO) << "Two descriptors are different.";
    }
  }

  // cv::Mat comparison;
  // cv::bitwise_xor(descriptors, cv_descriptors, comparison);
  // LOG(INFO) << "If the result is same, the number should be zero... : "
  //          << cv::countNonZero(comparison);

  // cv::Mat result_img;
  // DrawKeyPointsAsCircles(img, keypoints, result_img);
  // cv::imshow("Original Image", img);
  // cv::imshow("Result Image", result_img);
  // cv::waitKey(0);
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "OpenCV SIFT Test Started.";

  // 1. Load Images.
  cv::Mat img, gray_img;
  img = cv::imread(FLAGS_image_path);

  TestSiftFeatures(img);

  LOG(INFO) << "OpenCV SIFT Test Finished.";

  return 0;
}
