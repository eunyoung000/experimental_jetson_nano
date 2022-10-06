#ifndef PERCEPTION_QRCODE_RECOGNIZER_H
#define PERCEPTION_QRCODE_RECOGNIZER_H

#include <assert.h>
#include <set>
#include <vector>
#include <unordered_map>

#include "opencv2/core/core.hpp"
#include "zbar.h"

namespace perception {

// Class that detects the QR code region and recognizes the content in the QR code.
class QrCodeRecognizer {
 public:
  explicit QrCodeRecognizer(bool enable_debugging);
  // Detects the QR code regions and recognizes the content for each QR code.
  // The recognized QR codes are added to the qr_results list.
  void DetectAndRecognize(const cv::Mat& input_image, std::vector<std::pair<std::string, std::vector<cv::Point2f>>>* qr_results, cv::Mat* output);

 protected:
  std::vector<cv::Rect> GetRegionOfInterest(const cv::Mat& frame);
  void RunRecognition(const cv::Mat& frame, cv::Rect roi, 
    std::vector<std::pair<std::string, std::vector<cv::Point2f>>>* qr_results, cv::Mat* output);

  // Create a zbar reader
  zbar::ImageScanner scanner_;
  bool enable_debugging_;
  bool print_latency_;

  // Output res
};

}  // namespace perception

#endif  // PERCEPTION_QRCODE_RECOGNIZER_H