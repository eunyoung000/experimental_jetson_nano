#include "qrcode_recognizer.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace perception {

using zbar::Image;

namespace {

constexpr bool kDebuggingMode = false;
constexpr float kResizeRatio = 0.2f;

int elapsed_time_printing_count = 0;
// Renders the computation time to the output image.
void VisualizeElapsedTime(const std::string& label, const auto& begin_ms, const auto& end_ms, cv::Mat* output) {
    auto elapsed_ms = label + " " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_ms - begin_ms).count());
    // Print out the computation time in ms at the top corner of the image.
    const int text_x = static_cast<int>(output->cols * 0.05f);
    const int text_y = static_cast<int>(output->rows * 0.05f);
    putText(*output, elapsed_ms, cv::Point(text_x, text_y + elapsed_time_printing_count * 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 5);
    elapsed_time_printing_count++;
}

void ScaleBoundingBox(float scale, cv::Rect* bounding_box) {
    bounding_box->x *= scale;
    bounding_box->y *= scale;
    bounding_box->width *= scale;
    bounding_box->height *= scale;
}

void AddMarginToBoundingBox(float margin_ratio, cv::Rect* bounding_box) {
    const int x_margin = bounding_box->width * margin_ratio;
    const int y_margin = bounding_box->height * margin_ratio;
    bounding_box->x = std::max(0, bounding_box->x - x_margin);
    bounding_box->y = std::max(0, bounding_box->y - y_margin);
    bounding_box->width += 2 * x_margin;
    bounding_box->height += 2 * y_margin;
}

}  // namespace

QrCodeRecognizer::QrCodeRecognizer(bool enable_debugging) : enable_debugging_(enable_debugging) {
    // Configure the reader
    // Disable other symbols.
    scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
    // Enable the QR code recognition only.
    scanner_.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
    // High density QR code.
    scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, 1);
    scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, 1);
}

void GetQrCodeRegions(const zbar::Image& image, const cv::Rect& roi,
 std::vector<std::pair<std::string, std::vector<cv::Point2f>>>* qr_results) {
    for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
        // Draw location of the symbols found
        if (symbol->get_location_size() == 4) {
            std::vector<cv::Point2f> points(4);
            cv::Point2f mean_point(0.f, 0.f);
            for (int i = 0; i < 4; ++i) {
                points[i] = cv::Point2f(symbol->get_location_x(i) + roi.x, symbol->get_location_y(i) + roi.y);
                mean_point += points[i];
            }
            mean_point /= 4;
            std::sort(points.begin(), points.end(), [mean_point](const cv::Point2f& p1, const cv::Point2f& p2) {
                auto p1_vector = cv::Point(p1.x - mean_point.x, p1.y - mean_point.y);
                auto p2_vector = cv::Point(p2.x - mean_point.x, p2.y - mean_point.y);

                // Add "-" to convert from the image coordinate system.
                return atan2(-p1_vector.y, p1_vector.x) > atan2(-p2_vector.y, p2_vector.x);
            });
            qr_results->push_back(std::make_pair(symbol->get_data(), points));
        }
    }
}


void QrCodeRecognizer::RunRecognition(const cv::Mat& frame, cv::Rect roi,
   std::vector<std::pair<std::string, std::vector<cv::Point2f>>>* qr_results, cv::Mat* output) {
    cv::Mat frame_grayscale;
    cv::cvtColor(frame, frame_grayscale, cv::COLOR_BGR2GRAY);
    uchar *raw = (uchar *)(frame_grayscale.data);
    const int width = frame.cols;
    const int height = frame.rows;
    // Wrap image data
    Image image(width, height, "Y800", raw, width * height);

    // Scan the image for barcodes
    std::chrono::steady_clock::time_point begin_ms = std::chrono::steady_clock::now();
    scanner_.scan(image);
    std::chrono::steady_clock::time_point end_ms = std::chrono::steady_clock::now();

    GetQrCodeRegions(image, roi, qr_results);
    if (output != nullptr) {
        if (enable_debugging_) {
            VisualizeElapsedTime("QR code computation in ms", begin_ms, end_ms, output);
        }
    }

    // clean up
    image.set_data(NULL, 0);

    if (enable_debugging_) {
        cv::rectangle(*output, roi, cv::Scalar(255, 0, 0), 10);
        for (const auto& qr_result : *qr_results) {
            cv::Rect qr_region = cv::boundingRect(qr_result.second);
            cv::rectangle(*output, qr_region, cv::Scalar(0, 255, 0), 10);
        }
    }
}

void QrCodeRecognizer::DetectAndRecognize(const cv::Mat& input_image,
 std::vector<std::pair<std::string, std::vector<cv::Point2f>>>* qr_results, cv::Mat* output) {
    elapsed_time_printing_count = 0;
    auto begin_ms = std::chrono::steady_clock::now();
    const auto& rois = GetRegionOfInterest(input_image);
    auto end_ms = std::chrono::steady_clock::now();
    if (rois.empty()) {
        LOG(WARNING) << "No region of interest.";
        return;
    }
    
    if (enable_debugging_) {
        VisualizeElapsedTime("ROI computation in ms", begin_ms, end_ms, output);
    }

    for (auto& roi : rois) {
        const auto cropped = input_image(roi);

        LOG(INFO) << "image based roi resolution: " << cropped.cols << " " << cropped.rows;
        RunRecognition(cropped, roi, qr_results, output);
    }    
}

std::vector<cv::Rect> QrCodeRecognizer::GetRegionOfInterest(const cv::Mat& frame) {
    float scale = kResizeRatio;
    const int resized_width = static_cast<int>(scale * frame.cols);
    const int resized_height = static_cast<int>(scale * frame.rows);

    LOG(INFO) << "resized image size: " << resized_width << " "<<  resized_height;
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(resized_width, resized_height));

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::Mat resized_grayimage;
    cv::cvtColor(resized_frame, resized_grayimage, cv::COLOR_BGR2GRAY);

   // cv::threshold(resized_grayimage, resized_grayimage, 75, 255, cv::THRESH_BINARY_INV);
    cv::adaptiveThreshold(resized_grayimage, resized_grayimage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 20);
    //imshow("binary", resized_grayimage);

    // Erode to fill the small gap in the qrcode pattern (black part).
    cv::dilate(resized_grayimage, resized_grayimage, element, cv::Point(-1, -1), /*iterations=*/2, /*borderType=*/1, 1);
    // imshow("dilate", resized_grayimage);
    // Delete the noise pattern.
    // blur( resized_grayimage, resized_grayimage, Size(3,3) );
    // cv::threshold(resized_grayimage, resized_grayimage, 50, 255, cv::THRESH_BINARY_INV);
    // imshow("resized_grayimage", resized_grayimage);

    cv::erode(resized_grayimage, resized_grayimage, element, cv::Point(-1, -1), 
        /*iterations=*/3, /*borderType=*/1, 1);
  
    cv::imshow("erode", resized_grayimage);
     
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(resized_grayimage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // imshow("resized_grayimage", resized_grayimage);
    std::vector<cv::Rect> rois;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect bounding_box = boundingRect(contours[i]);
        // vector<Point> contours_poly;
        // float perimeter = arcLength(contours[i], true);
        // approxPolyDP(contours[i], contours_poly, /*epsilon=*/0.05 * perimeter, /*closed=*/true);
        // if (contours_poly.size() != 4) {
        //     std::cout << "Contour poly size is not 4 " << contours_poly.size() << std::endl;
        //     continue;
        // }
        
        float ratio = bounding_box.width / static_cast<float>(bounding_box.height);

        if (ratio < 0.5 || ratio > 1.5) {
            // std::cout << "Box ratio isn't matched with the condition " << ratio << std::endl;
            continue;
        }
        // Tranform the box back to the original coordinate system.
        ScaleBoundingBox(1.f / scale, &bounding_box);
        AddMarginToBoundingBox(0.15f, &bounding_box);
   
        // Filter if the region is smaller than 100x100 or bigger than 500x500.
        if (bounding_box.width < 50 || bounding_box.height < 50) {
            // std::cout << "The size of the box is too small" << bounding_box.width << " " << bounding_box.height << std::endl;
            continue;
        }
        if (bounding_box.width > 300 || bounding_box.height > 300) {
            // std::cout << "The size of the box is too large" << bounding_box.width << " " << bounding_box.height << std::endl;
            continue;
        }
        if (bounding_box.x + bounding_box.width >= frame.cols || bounding_box.y + bounding_box.height >= frame.rows) {
            continue;
        }
        LOG(INFO) << "bounding box width & height " << bounding_box.width << " " << bounding_box.height;
        rois.push_back(bounding_box);
    }
    return rois;
}

}  // namespace perception