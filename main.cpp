#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_map>
#include <vector>

#include "box_fitting.h"
#include "focuser.h"
#include "opencv2/core/core.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include "zbar.h"

using namespace std;
using namespace cv;
using namespace zbar;

namespace {

// Maximum width/height: 4056, 3040.
// 2028, 1520
constexpr int kImageWidth = 1920; // 4032;
constexpr int kImageHeight = 1080; // 3040;
constexpr bool kDebuggingMode = false;
constexpr float kResizeRatio = 0.2f;

std::string GetGStreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method=0) {
    return "nvarguscamerasrc aelock=true gainrange=\"7 7\" exposuretimerange=\"5000000 5000000\" !  video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// std::string GetGStreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method=0) {
//     return "v4l2src device=/dev/video0 ! image/jpeg !jpegdec !video/x-raw, format=I420 ! videoconvert ! appsink";
// }

int elapsed_time_printing_count = 0;
// Renders the computation time to the output image.
void VisualizeElapsedTime(const string& label, const auto& begin_ms, const auto& end_ms, cv::Mat* output) {
    auto elapsed_ms = label + " " + to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_ms - begin_ms).count());
    // Print out the computation time in ms at the top corner of the image.
    const int text_x = static_cast<int>(output->cols * 0.05f);
    const int text_y = static_cast<int>(output->rows * 0.05f);
    putText(*output, elapsed_ms, cv::Point(text_x, text_y + elapsed_time_printing_count * 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 5);
    elapsed_time_printing_count++;
}


class ZbarQrCodeRecognizer {
 public:
  ZbarQrCodeRecognizer() {
    // Configure the reader
    // Disable other symbols.
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 0);
    // Enable the QR code recognition only.
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
    // High density QR code.
    scanner.set_config(zbar::ZBAR_NONE, ZBAR_CFG_X_DENSITY, 1);
    scanner.set_config(zbar::ZBAR_NONE, ZBAR_CFG_Y_DENSITY, 1);
  }

  void GetQrCodeRegions(const Image& image, const cv::Rect& roi, std::vector<std::pair<std::string, std::vector<cv::Point2f>>>* qr_results) {
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

  void RunRecognition(const cv::Mat& frame, cv::Rect roi, std::vector<std::pair<std::string, std::vector<cv::Point2f>>>* qr_results, cv::Mat* output) {
    cv::Mat frame_grayscale;
    cvtColor(frame, frame_grayscale, cv::COLOR_BGR2GRAY);
    uchar *raw = (uchar *)(frame_grayscale.data);
    const int width = frame.cols;
    const int height = frame.rows;
    // Wrap image data
    Image image(width, height, "Y800", raw, width * height);

    // Scan the image for barcodes
    std::chrono::steady_clock::time_point begin_ms = std::chrono::steady_clock::now();
    scanner.scan(image);
    std::chrono::steady_clock::time_point end_ms = std::chrono::steady_clock::now();
    
    GetQrCodeRegions(image, roi, qr_results);
    if (output != nullptr) {
        if (kDebuggingMode) {
            VisualizeElapsedTime("QR code computation in ms", begin_ms, end_ms, output);
        }
    }

    // clean up
    image.set_data(NULL, 0);
  }
 private:
    // Create a zbar reader
    ImageScanner scanner;
};

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

std::vector<cv::Rect> GetRegionOfInterest(const Mat& frame) {
    float scale = kResizeRatio;
    const int resized_width = static_cast<int>(scale * frame.cols);
    const int resized_height = static_cast<int>(scale * frame.rows);

    cout << "resized image size: " << resized_width << " "<<  resized_height << endl;
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(resized_width, resized_height));

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::Mat resized_grayimage;
    cv::cvtColor(resized_frame, resized_grayimage, cv::COLOR_BGR2GRAY);

   // cv::threshold(resized_grayimage, resized_grayimage, 75, 255, cv::THRESH_BINARY_INV);
    cv::adaptiveThreshold(resized_grayimage, resized_grayimage, 255, ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV, 21, 20);
    //imshow("binary", resized_grayimage);

    // Erode to fill the small gap in the qrcode pattern (black part).
    dilate(resized_grayimage, resized_grayimage, element, Point(-1, -1), /*iterations=*/2, /*borderType=*/1, 1);
    // imshow("dilate", resized_grayimage);
    // Delete the noise pattern.
    // blur( resized_grayimage, resized_grayimage, Size(3,3) );
    // cv::threshold(resized_grayimage, resized_grayimage, 50, 255, cv::THRESH_BINARY_INV);
    // imshow("resized_grayimage", resized_grayimage);

    erode(resized_grayimage, resized_grayimage, element, Point(-1, -1), 
        /*iterations=*/3, /*borderType=*/1, 1);
  
    imshow("erode", resized_grayimage);
     
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
        std::cout << "bounding box width & height " << bounding_box.width << " " << bounding_box.height << std::endl;
        rois.push_back(bounding_box);
    }
    return rois;
}

void DisplayOutput(const Mat& output_image, float scale) {
    const int resized_width = static_cast<int>(scale * output_image.cols);
    const int resized_height = static_cast<int>(scale * output_image.rows);
    cv::Mat resized_image;
    cv::resize(output_image, resized_image, Size(resized_width, resized_height));
    imshow("Captured frame", resized_image);
}

VideoCapture GetVideoCapture(const string& video_path) {
    if (video_path.empty()) {
        return VideoCapture(
            GetGStreamerPipeline(/*capture_width=*/kImageWidth, /*capture_height=*/kImageHeight,
             /*display_width=*/kImageWidth, /*display_height=*/kImageHeight, /*frame_rate=*/10), cv::CAP_GSTREAMER);
    }
    return VideoCapture(video_path);
}

std::unordered_map<std::string, perception::SKUInfo> IntializeModelInfo() {
    std::unordered_map<std::string, perception::SKUInfo> model_info;
    const std::string kFirstSKUId = "sku 1234567890";
    perception::SKUInfo first_sku_info(kFirstSKUId, 279.4f, 152.4f, 152.4f);
    first_sku_info.SetWeight(3);
    // model_info[kFirstSKUId].width_mm = 279.4;
    // model_info[sku_id].height_mm = 152.4;
    // model_info[sku_id].depth_mm = 152.4;
    // model_info[sku_id].weight_kg = 3;
    // model_info[sku_id].Create3DBoxVertices();
    // Not measured. This value is approximately given.
    *first_sku_info.GetMutableFeatureVertices() = {
        { 80, 118.9, 152.4f },
        { 168.9, 118.9, 152.4f },
        { 168.9, 30, 152.4f },
        { 80, 30, 152.4f }
    };
    model_info[kFirstSKUId] = first_sku_info;

    // model_info[sku_id].sku_id = kFirstSKUId;

    // Should read this from the file or server.
    // Big box.
    // model_info[sku_id].object_vertices = {
    //     {0, 0, 0},
    //     {0, 461.96, 0},
    //     {719.13, 461.96, 0},
    //     {719.13, 0, 0},
    //     {0, 0, 790.575},
    //     {0, 461.96, 790.575},
    //     {719.13, 461.96, 790.575},
    //     {719.13, 0, 790.575} };

    // model_info[sku_id].qr_vertices = {
    //     {293.6875, 396.88, 790.575},
    //     {382.58, 396.88, 790.575},
    //     {382.58, 307.98, 790.575},
    //     {293.6875, 307.98, 790.575}
    // };
    // model_info[sku_id].width_mm = 719.13;
    // model_info[sku_id].height_mm = 790.575;
    // model_info[sku_id].depth_mm = 461.96;
    // model_info[sku_id].weight_kg = 10;


    // Another QR object.

    // std::string sku_id_2 = "sku 485763";
    // // Should read this from the file or server.
    // model_info[sku_id_2].width_mm = 279.4;
    // model_info[sku_id_2].height_mm = 152.4;
    // model_info[sku_id_2].depth_mm = 152.4;
    // model_info[sku_id_2].weight_kg = 3;
    // model_info[sku_id_2].Create3DBoxVertices();
    // // Not measured. This value is approximately given.
    // float qrcode_size = 88.9;
    // float min_y = 32;
    // float min_x = 90;
    // model_info[sku_id_2].qr_vertices = {
    //     {min_x, min_y + qrcode_size, model_info[sku_id_2].height_mm },
    //     {min_x + qrcode_size, min_y + qrcode_size, model_info[sku_id_2].height_mm },
    //     {min_x + qrcode_size, min_y, model_info[sku_id_2].height_mm },
    //     {min_x, min_y, model_info[sku_id_2].height_mm }
    // };

    // model_info[sku_id_2].sku_id = sku_id_2;

    const std::string kSecondSKUId = "sku 485763";
    perception::SKUInfo second_sku_info(kSecondSKUId, 279.4f, 152.4f, 152.4f);
    second_sku_info.SetWeight(3);
    // model_info[kFirstSKUId].width_mm = 279.4;
    // model_info[sku_id].height_mm = 152.4;
    // model_info[sku_id].depth_mm = 152.4;
    // model_info[sku_id].weight_kg = 3;
    // model_info[sku_id].Create3DBoxVertices();
    // Not measured. This value is approximately given.
    *second_sku_info.GetMutableFeatureVertices() = {
        { 80, 118.9, 152.4f },
        { 168.9, 118.9, 152.4f },
        { 168.9, 30, 152.4f },
        { 80, 30, 152.4f }
    };

    float qrcode_size = 88.9;
    float min_y = 32;
    float min_x = 90;
    *second_sku_info.GetMutableFeatureVertices() = {
        {min_x, min_y + qrcode_size, 152.4f },
        {min_x + qrcode_size, min_y + qrcode_size, 152.4f },
        {min_x + qrcode_size, min_y, 152.4f },
        {min_x, min_y, 152.4f }
    };
    model_info[kSecondSKUId] = second_sku_info;

    return model_info;
}

// void AddNewBoxToTracker(const cv::Rect& roi, MultiTracker* tracker) {
//     multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(bboxes[i]));
// }

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

// int main(int argc, char **argv) {
//     string video_path = "";
//     if (argc == 2) {
//         // If a video path is given as an input, it processes the video.
//         video_path = argv[1];
//     }
//     // Create a potential loading region on the forklift.
//     float rotation_vector[3] = {2.561808810508662, -0.01263617213322713, -0.004604882284415327};
//     float translation_vector[3] = {-791.8981534713707, 710.0323401713475, 1562.565190955901};
//     perception::ModelInfo forklift(/*width_mm=*/1000.f, /*height_mm=*/ 2000.f, /*depth_mm=*/ 1000.f, rotation_vector, translation_vector);
//     *forklift.GetMutableFeatureVertices() = {
//         {700, 538.9, 0},
//         {788.9, 538.9, 0},
//         {788.9, 450, 0},
//         {700, 450, 0}
//     };

//     // Set up the camera intrinsic parameters.
//     cv::Mat camera_matrix(3,3,cv::DataType<float>::type);
//     cv::setIdentity(camera_matrix);
//     camera_matrix.at<float>(0, 0) = 3806.59493;
//     camera_matrix.at<float>(0, 2) = 2143.26588;
//     camera_matrix.at<float>(1, 1) = 3785.49799;
//     camera_matrix.at<float>(1, 2) = 1453.25031;
    
//     // std::cout << "camera matrix: " << camera_matrix << std::endl;

//     cv::Mat distortion_coefficients(5,1,cv::DataType<float>::type);
//     distortion_coefficients.at<float>(0) = 0.22491184;
//     distortion_coefficients.at<float>(1) = -0.93347666;
//     distortion_coefficients.at<float>(2) = -0.00787728;
//     distortion_coefficients.at<float>(3) = 0.00867964;
//     distortion_coefficients.at<float>(4) = 1.47307115;

//     std::unique_ptr<perception::BoxFitting> box_fitting = std::make_unique<perception::BoxFitting>(camera_matrix, distortion_coefficients);

//     perception::ModelInfo qr_model(/*width_mm=*/88.9f, /*height_mm=*/ 0.f, /*depth_mm=*/ 88.9f);
//     // qr_model.width_mm = 88.9;
//     // qr_model.height_mm = 0.f;
//     // qr_model.depth_mm = 88.9f;
//     // qr_model.object_vertices = {
//     //     {0, 88.9, 0},
//     //     {88.9, 88.9, 0},
//     //     {88.9, 0, 0},
//     //     {0, 0, 0}
//     //};

    
//     // Load the model information.
//     std::unordered_map<std::string, perception::SKUInfo> model_info = IntializeModelInfo();
//    // model_info["sku 1234567890"] = forklift;

//     VideoCapture capture = GetVideoCapture(video_path);
//     const int kInitialFocusValue = 0;
//     if (!capture.isOpened()) {
//         cerr << "Could not open camera." << endl;
//         exit(EXIT_FAILURE);
//     }
//     namedWindow("Captured frame");
    
//     camera::Focuser focuser(/*bus_info=*/7, kInitialFocusValue);

//     // YoloQrCodeDetector yolo_qr_detector;
//     ZbarQrCodeRecognizer zbar_qr_recognizer;
    
//     // Capture an OpenCV frame
//     cv::Mat frame, old_gray_frame, original_frame;
//     int num_consecutive_missing_frames = 0;
//     cv::Rect focus_roi(1000, 500, 2000, 2000);
    
//     std::vector<cv::Point2f> old_points;

//     while (true) {
//         capture >> original_frame;
//         if (original_frame.empty()) {
//             break;
//         }
//         if (video_path.empty()) {
//             // Run auto focus when the input is from the camera.
//             auto focused = focuser.Focusing(original_frame); //, focus_roi);
//             if (!focused) {
//                 continue;
//             }
//         }
//         cv::Mat output_image = original_frame.clone();
//         cv::Mat resized_frame;
//         cv::resize(original_frame, resized_frame, cv::Size(403, 304));
//         cv::Mat resized_grayimage;
//         cv::cvtColor(resized_frame, resized_grayimage, cv::COLOR_BGR2GRAY);

//         std::vector<cv::Rect> tracked_rois;
//         if (!old_gray_frame.empty() && old_points.size() > 0) {
//             cv::TermCriteria criteria = cv::TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
//             std::vector<cv::Point2f> good_new, current_points;

//             std::vector<uchar> status;
//             std::vector<float> err;
//             cv::calcOpticalFlowPyrLK(old_gray_frame, resized_grayimage, old_points, current_points, status, err, cv::Size(15,15), 2, criteria);     
//             std::cout << "### old_points " << old_points.size() << std::endl;
//             const int num_tracked_boxes = (int) old_points.size() / 4;
//             for(int i = 0; i < num_tracked_boxes; ++i) {
//                 std::vector<cv::Point2f> corners;
//                 int valid_count = 0;
//                 for (int j = 0; j < 4; j++) {
//                     if (status[i * 4 + j] == 1) {
//                         valid_count++;
//                     }
//                     corners.push_back(current_points[i * 4 + j]);
//                 }
//                 if (valid_count >= 3) {
//                     for (int j = 0; j < 4; j++) {
//                         good_new.push_back(corners[j]);
//                         // draw the tracks
//                         // Bring the points back to the original coordinate system.
//                         corners[j] = corners[j] * 10;
//                         cv::circle(output_image, corners[j], 20, cv::Scalar(0, 0, 255), -1);
//                     }
//                     cv::Rect bounding_box = cv::boundingRect(corners);
//                     AddMarginToBoundingBox(0.1f, &bounding_box);
//                     tracked_rois.push_back(bounding_box);
//                 }
                
//             }
//             std::cout << "### good_new " << good_new.size() << std::endl;
//             old_points = good_new;   
//         }


//         // cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
        
//         std::vector<std::pair<std::string, std::vector<cv::Point2f>>> qr_results;
//         for (const auto& roi: tracked_rois) {
//             const auto cropped = original_frame(roi);
//            // Mat bbox, rectified_image;
//            // auto decoded_string = qrDecoder.detectAndDecode(cropped, bbox, rectified_image);
//            // cout <<"opencv output " << bbox.cols << " " << bbox.rows << " " << decoded_string << endl;
//            //  imshow("cropped", cropped);

//             cout << "roi resolution: " << cropped.cols << " " << cropped.rows << endl;
//             zbar_qr_recognizer.RunRecognition(cropped, roi, &qr_results, &output_image);
//             if (kDebuggingMode) {
//                 cv::rectangle(output_image, roi, cv::Scalar(255, 0, 0), 10);
//             }

//         }

//         elapsed_time_printing_count = 0;
//         auto begin_ms = std::chrono::steady_clock::now();

//         const auto& rois = GetRegionOfInterest(original_frame);
//         auto end_ms = std::chrono::steady_clock::now();
//         if (rois.empty()) {
//             continue;
//         }
       
//         if (kDebuggingMode) {
//             VisualizeElapsedTime("ROI computation in ms", begin_ms, end_ms, &output_image);
//         }

//         int index_from_new_rois = qr_results.size();
//         for (auto& roi : rois) {
//             if (IsAlreadyBeingTracked(roi, tracked_rois)) {
//                 std::cout << "Skipped! " << std::endl;
//                 continue;
//             }
//             const auto cropped = original_frame(roi);
//         // Mat bbox, rectified_image;
//         // auto decoded_string = qrDecoder.detectAndDecode(cropped, bbox, rectified_image);
//         // cout <<"opencv output " << bbox.cols << " " << bbox.rows << " " << decoded_string << endl;
//         //  imshow("cropped", cropped);

//             cout << "image based roi resolution: " << cropped.cols << " " << cropped.rows << endl;
//             zbar_qr_recognizer.RunRecognition(cropped, roi, &qr_results, &output_image);
//             if (kDebuggingMode) {
//                 cv::rectangle(output_image, roi, cv::Scalar(255, 0, 0), 10);
//             }
//         }
//         std::cout << "##### " << index_from_new_rois << " "<< qr_results.size() << std::endl;
//         if (qr_results.size() > index_from_new_rois) { // && track_id == -1) {
//             for (int roi_index = index_from_new_rois; roi_index < qr_results.size(); roi_index++) {
//                 for (int i = 0; i < qr_results[roi_index].second.size(); ++i) {
//                     cv::Point2f tmp = qr_results[roi_index].second[i] * 0.1f;
//                     old_points.push_back(tmp);
//                 }
//             }
//         }

//         box_fitting->Run3DBoxFitting(original_frame, qr_results, qr_model, forklift, model_info, &output_image);

//         // Show captured frame, now with overlays!
//         DisplayOutput(output_image, 0.5f);
                                                                                                                                                          
//         int key = waitKey(100) & 0xff;
            
//         // Stop the program on the ESC key.
//         if (key == 27) {
//             break;
//         }
//         old_gray_frame = resized_grayimage.clone();
//     }
//     capture.release();
//     return 0;
// }

int main(int argc, char **argv) {
    string video_path = "";
    if (argc == 2) {
        // If a video path is given as an input, it processes the video.
        video_path = argv[1];
    }
    // Create a potential loading region on the forklift.
    float rotation_vector[3] = {2.561808810508662, -0.01263617213322713, -0.004604882284415327};
    float translation_vector[3] = {-791.8981534713707, 710.0323401713475, 1562.565190955901};
    perception::ModelInfo forklift(/*width_mm=*/1000.f, /*height_mm=*/ 2000.f, /*depth_mm=*/ 1000.f, rotation_vector, translation_vector);
    *forklift.GetMutableFeatureVertices() = {
        {700, 538.9, 0},
        {788.9, 538.9, 0},
        {788.9, 450, 0},
        {700, 450, 0}
    };

    // Set up the camera intrinsic parameters.
    cv::Mat camera_matrix(3, 3,cv::DataType<float>::type);
    cv::setIdentity(camera_matrix);
    camera_matrix.at<float>(0, 0) = 3806.59493;
    camera_matrix.at<float>(0, 2) = 2143.26588;
    camera_matrix.at<float>(1, 1) = 3785.49799;
    camera_matrix.at<float>(1, 2) = 1453.25031;
    
    // std::cout << "camera matrix: " << camera_matrix << std::endl;

    cv::Mat distortion_coefficients(5,1,cv::DataType<float>::type);
    distortion_coefficients.at<float>(0) = 0.22491184;
    distortion_coefficients.at<float>(1) = -0.93347666;
    distortion_coefficients.at<float>(2) = -0.00787728;
    distortion_coefficients.at<float>(3) = 0.00867964;
    distortion_coefficients.at<float>(4) = 1.47307115;

    std::unique_ptr<perception::BoxFitting> box_fitting = std::make_unique<perception::BoxFitting>(camera_matrix, distortion_coefficients);

    perception::ModelInfo qr_model(/*width_mm=*/88.9f, /*height_mm=*/ 0.f, /*depth_mm=*/ 88.9f);
    // qr_model.width_mm = 88.9;
    // qr_model.height_mm = 0.f;
    // qr_model.depth_mm = 88.9f;
    // qr_model.object_vertices = {
    //     {0, 88.9, 0},
    //     {88.9, 88.9, 0},
    //     {88.9, 0, 0},
    //     {0, 0, 0}
    //};

    
    // Load the model information.
    std::unordered_map<std::string, perception::SKUInfo> model_info = IntializeModelInfo();
   // model_info["sku 1234567890"] = forklift;

    VideoCapture capture = GetVideoCapture(video_path);
    const int kInitialFocusValue = 130;
    if (!capture.isOpened()) {
        cerr << "Could not open camera." << endl;
        exit(EXIT_FAILURE);
    }
    namedWindow("Captured frame");
    
    camera::Focuser focuser(/*bus_info=*/7, kInitialFocusValue);

    // YoloQrCodeDetector yolo_qr_detector;
    ZbarQrCodeRecognizer zbar_qr_recognizer;
    
    // Capture an OpenCV frame
    cv::Mat frame, old_gray_frame, original_frame;
    int num_consecutive_missing_frames = 0;

    std::vector<std::pair<std::string, std::vector<cv::Point2f>>> qr_results;
    while (true) {
        capture >> original_frame;
        if (original_frame.empty()) {
            break;
        }
        // if (video_path.empty()) {
        //     // Run auto focus when the input is from the camera.
        //     auto focused = focuser.Focusing(original_frame); //, focus_roi);
        //     if (!focused) {
        //         continue;
        //     }
        // }

        std::cout <<"## original size " << original_frame.cols << " " << original_frame.rows << std::endl;
        cv::Mat output_image = original_frame.clone();
        cv::Mat resized_frame;
        cv::resize(original_frame, resized_frame, cv::Size(403, 304));
        cv::Mat resized_grayimage;
        cv::cvtColor(resized_frame, resized_grayimage, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> tracked_rois;
        if (!old_gray_frame.empty() && qr_results.size() > 0) {
            cv::TermCriteria criteria = cv::TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
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
                        cv::circle(output_image, corners[j], 20, cv::Scalar(0, 0, 255), -1);
                    }
                    cv::Rect bounding_box = cv::boundingRect(corners);
                    tracked_rois.push_back(bounding_box);
                    new_qr_results.push_back(qr_results[i]);
                }
                
            }
            //std::cout << "#### " << qr_results.size() << " " << new_qr_results.size() << std::endl;
            qr_results = new_qr_results;
            //std::cout << "### good_new " << good_new.size() << std::endl;
            old_points = good_new;   
        }

        elapsed_time_printing_count = 0;
        auto begin_ms = std::chrono::steady_clock::now();

        const auto& rois = GetRegionOfInterest(original_frame);
        auto end_ms = std::chrono::steady_clock::now();
        if (rois.empty()) {
            continue;
        }
       
        if (kDebuggingMode) {
            VisualizeElapsedTime("ROI computation in ms", begin_ms, end_ms, &output_image);
        }

        for (auto& roi : rois) {
            if (IsAlreadyBeingTracked(roi, tracked_rois)) {
                // std::cout << "Skipped! " << std::endl;
                continue;
            }
            const auto cropped = original_frame(roi);
        // Mat bbox, rectified_image;
        // auto decoded_string = qrDecoder.detectAndDecode(cropped, bbox, rectified_image);
        // cout <<"opencv output " << bbox.cols << " " << bbox.rows << " " << decoded_string << endl;
        //  imshow("cropped", cropped);

            cout << "image based roi resolution: " << cropped.cols << " " << cropped.rows << endl;
            zbar_qr_recognizer.RunRecognition(cropped, roi, &qr_results, &output_image);
            if (true) {
                cv::rectangle(output_image, roi, cv::Scalar(255, 0, 0), 10);
            }
        }
        // box_fitting->Run3DBoxFitting(original_frame, qr_results, qr_model, forklift, model_info, &output_image);

        // Show captured frame, now with overlays!
        DisplayOutput(output_image, 0.5f);
                                                                                                                                                          
        int key = waitKey(100) & 0xff;
            
        // Stop the program on the ESC key.
        if (key == 27) {
            break;
        }
        old_gray_frame = resized_grayimage.clone();
    }
    capture.release();
    return 0;
}