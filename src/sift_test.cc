

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

void DrawKeyPointsAsCircles(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints,
                            cv::Mat& result) {
  img.copyTo(result);
  for (auto pnt : keypoints) {
    cv::Point2i location(pnt.pt.x + 0.5, pnt.pt.y + 0.5);
    cv::circle(result, location, static_cast<int>(pnt.size + 0.5f), cv::Scalar(0, 0, 255));
  }
}

void TestSiftFeatures(cv::Mat& img) {
  cv_copy::SIFT sift_detector;

  std::vector<cv::KeyPoint> keypoints;
  sift_detector.Detect(img, keypoints);

  cv::Mat descriptors;
  sift_detector.Compute(img, keypoints, descriptors);

  cv::Mat result_img;
  DrawKeyPointsAsCircles(img, keypoints, result_img);

  cv::imshow("Original Image", img);
  cv::imshow("Result Image", result_img);
  cv::waitKey(0);

  LOG(INFO) << keypoints.size();
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
