

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
#include "xfeatures2d_copy.hpp"

using namespace cv_copy;

DEFINE_string(image_path, "", "Path to the image.");

void LoadImages(const std::string& image_path, cv::Mat& img, cv::Mat& gray_img) {
  img = cv::imread(image_path);
  cv::imshow("Sample Image", img);
  cv::waitKey(0);
}

void TestSiftFeatures(cv::Mat& img) {
  std::vector<cv::KeyPoint> keypoints;

  cv_copy::SIFT sift_detector;

  sift_detector.detect(img, keypoints);

  LOG(INFO) << keypoints.size();
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Vlfeat SIFT Test Started.";

  // 1. Load Images.
  cv::Mat img, gray_img;
  LoadImages(FLAGS_image_path, img, gray_img);

  TestSiftFeatures(img);

  LOG(INFO) << "Vlfeat SIFT Test Finished.";

  return 0;
}
