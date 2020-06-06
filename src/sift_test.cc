

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
#include "sift_from_opencv.hpp"

using namespace cv_copy;

DEFINE_string(image_path, "", "Path to the image.");

namespace {

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
    cv::circle(result, location, static_cast<int>(pnt.size + 0.5f), cv::Scalar(0, 255, 255), 2);
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

void TestSiftFeatures(const std::string& image_path) {
  // cv::setNumThreads(0);

  cv::Mat img = cv::imread(image_path);

  std::vector<SiftKeyPoint> keypoints;
  std::vector<cv::KeyPoint> cv_keypoints;
  {
    // Run refactored SIFT detector.
    cv_copy::SIFT sift_detector;
    sift_detector.Detect(img, keypoints);

    // Run original SIFT detector.
    cv::Ptr<cv_copy::xfeatures2d::SIFT> cv_sift_detector = cv_copy::xfeatures2d::SIFT::create();
    cv_sift_detector->detect(img, cv_keypoints);

    cv::Mat test;
    DrawKeyPointsAsCircles(img, keypoints, test);
    cv::resize(test, test, cv::Size(), 0.5, 0.5);
    cv::imshow("Test", test);
    cv::waitKey(0);

    // Check if the results are same.
    if (!KeyPointsAreSame(cv_keypoints, keypoints)) {
      LOG(INFO) << "Two keypoints are different.";
      return;
    }

    LOG(INFO) << "No of detected keypoints : " << keypoints.size();
    LOG(INFO) << "Two keypoints are same.";
  }

  cv::Mat cv_descriptors, descriptors;
  {
    // Run refactored sift descriptor.
    cv_copy::SIFT sift_detector;
    sift_detector.Compute(img, keypoints, descriptors);

    // Run original sift descriptor.
    cv::Ptr<cv_copy::xfeatures2d::SIFT> cv_sift_detector = cv_copy::xfeatures2d::SIFT::create();
    cv_sift_detector->compute(img, cv_keypoints, cv_descriptors);

    // Check if the results are same.
    if (!DescriptorsAreSame(descriptors, cv_descriptors)) {
      LOG(INFO) << "Two descriptors are different.";
      return;
    }
    LOG(INFO) << "Two descriptors are same.";
  }
}
}  // namespace

// Value defined in CMakeLists.txt file.
static const std::string project_folder_path = PRJ_FOLDER_PATH;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  std::string image_path = FLAGS_image_path;
  if (FLAGS_image_path == "") {
    image_path = project_folder_path + "/data/sample.jpg";
  }

  LOG(INFO) << "OpenCV SIFT Test Started.";

  TestSiftFeatures(image_path);

  LOG(INFO) << "OpenCV SIFT Test Finished.";

  return 0;
}
