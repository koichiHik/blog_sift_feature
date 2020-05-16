

#include <iomanip>
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Eigen
#include <eigen3/Eigen/Core>

// GFLAGS
#include <gflags/gflags.h>

// GLOG
#include <glog/logging.h>

// Original
#include "sift.hpp"
#include "xfeatures2d_copy_org.hpp"

using namespace cv_copy;

DEFINE_string(image_path, "", "Path to the image.");

static void DisplayImage(const cv::Mat& img, const std::string& title, const int wait_time) {
  double min_val, max_val;
  cv::minMaxLoc(img, &min_val, &max_val);
  cv::Mat tmp;

  img.convertTo(tmp, CV_8UC1, 255.0 / (max_val - min_val), -255.0 * min_val / (max_val - min_val));
  cv::minMaxLoc(tmp, &min_val, &max_val);
  cv::imshow(title, tmp);
  cv::waitKey(wait_time);
}

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

    LOG(INFO) << "No of detected keypoints : " << keypoints.size();
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
