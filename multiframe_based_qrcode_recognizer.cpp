#include "multiframe_based_qrcode_recognizer.h"

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
#include "opencv2/video.hpp"

namespace perception {

namespace {

bool IsAlreadyBeingTracked(const cv::Rect& roi, const std::vector<cv::Rect>& tracked_rois) {
    for(const auto& tracked_roi : tracked_rois) {
        float intersection_area = static_cast<float>((tracked_roi & roi).area());
        float union_area = static_cast<float>((tracked_roi | roi).area());
        if (intersection_area / union_area > 0.5) {
            return true;
        }
    }
    return false;
}

}  // namespace

void MultiFrameBasedQrCodeRecognizer::DetectAndRecognize(const cv::Mat& input_image, cv::Mat* output) {
    cv::Mat resized_frame;
    cv::resize(input_image, resized_frame, cv::Size(403, 304));
    cv::Mat resized_grayimage;
    cv::cvtColor(resized_frame, resized_grayimage, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> tracked_rois;
    if (!old_gray_frame.empty() && qr_results.size() > 0) {
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        std::vector<cv::Point2f> old_points, good_new, current_points;

        for (int roi_index = 0; roi_index < qr_results.size(); roi_index++) {
            for (int i = 0; i < qr_results[roi_index].second.size(); ++i) {
                cv::Point2f tmp = qr_results[roi_index].second[i] * 0.1f;
                old_points.push_back(tmp);
            }
        }
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(old_gray_frame, resized_grayimage, old_points, current_points, status, err, cv::Size(15,15), 2, criteria);     
        // std::cout << "### old_points " << old_points.size() << std::endl;
        const int num_tracked_boxes = (int) qr_results.size();

        std::vector<std::pair<std::string, std::vector<cv::Point2f>>> new_qr_results;
        for(int i = 0; i < num_tracked_boxes; ++i) {
            std::vector<cv::Point2f> corners;
            int valid_count = 0;
            for (int j = 0; j < 4; j++) {
                if (status[i * 4 + j] == 1 and err[i * 4 + j] < 10) {
                    valid_count++;
                }
                corners.push_back(current_points[i * 4 + j]);
            }
            // std::cout << "#### valid_count " << valid_count << std::endl;
            if (valid_count >= 4) {
                for (int j = 0; j < 4; j++) {
                    good_new.push_back(corners[j]);
                    // draw the tracks
                    // Bring the points back to the original coordinate system.
                    corners[j] = corners[j] * 10;
                    cv::circle(*output, corners[j], 20, cv::Scalar(0, 0, 255), -1);
                }
                cv::Rect bounding_box = cv::boundingRect(corners);
                tracked_rois.push_back(bounding_box);
                new_qr_results.push_back(qr_results[i]);
            }
            
        }
        qr_results = new_qr_results;
        old_points = good_new;   
    }
    const auto rois = GetRegionOfInterest(input_image);

    if (rois.empty()) {
        LOG(WARNING) << "No region of interest.";
        return;
    }
    for (const auto& roi : rois) {
        if (IsAlreadyBeingTracked(roi, tracked_rois)) {
            // std::cout << "Skipped! " << std::endl;
            continue;
        }
        const auto cropped = input_image(roi);

        LOG(INFO) << "image based roi resolution: " << cropped.cols << " " << cropped.rows;
        RunRecognition(cropped, roi, &qr_results, output);
    }  
    old_gray_frame = resized_grayimage.clone();
}

}  // namespace perception