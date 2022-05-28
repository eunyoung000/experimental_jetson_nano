#ifndef PERCEPTION_MULTIFRAME_BASED_QRCODE_RECOGNIZER_H
#define PERCEPTION_MULTIFRAME_BASED_QRCODE_RECOGNIZER_H

#include <assert.h>
#include <set>
#include <vector>
#include <unordered_map>

#include "opencv2/core/core.hpp"
#include "zbar.h"

#include "qrcode_recognizer.h"

namespace perception {

// Class that detects the QR code region and recognizes the content in the QR code.
class MultiFrameBasedQrCodeRecognizer: public QrCodeRecognizer {
 public:
  MultiFrameBasedQrCodeRecognizer(bool enable_debugging) : QrCodeRecognizer(enable_debugging) {}
  void DetectAndRecognize(const cv::Mat& input_image, cv::Mat* output);
 
 private:
  void Reset();
  cv::Mat old_gray_frame;
  std::vector<std::pair<std::string, std::vector<cv::Point2f>>> qr_results;
};

}  // namespace perception

#endif  // PERCEPTION_MULTIFRAME_BASED_QRCODE_RECOGNIZER_H