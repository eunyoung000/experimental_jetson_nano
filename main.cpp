#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_map>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "opencv2/core/core.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>

#include "box_fitting.h"
#include "focuser.h"
#include "multiframe_based_qrcode_recognizer.h"
#include "qrcode_recognizer.h"

namespace {

DEFINE_bool(enable_debugging, false, "True to visualize the intermediate results for debugging");
DEFINE_bool(enable_tracking, false, "True to enable tracking");
DEFINE_bool(enable_3d_box_fitting, false, "True to enable 3d box fitting");
DEFINE_int32(image_width, 1920, "Image width up to 4032");
DEFINE_int32(image_height, 1080, "Image height up to 3040");
DEFINE_bool(visualize_boxes, false, "True to visualize the detected boxes in the image");
DEFINE_string(video_path, "", "Run the process with a saved video");

std::string GetGStreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method=0) {
    return "nvarguscamerasrc aelock=true gainrange=\"7 7\" exposuretimerange=\"5000000 5000000\" !  video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


void DisplayOutput(const cv::Mat& output_image, float scale) {
    const int resized_width = static_cast<int>(scale * output_image.cols);
    const int resized_height = static_cast<int>(scale * output_image.rows);
    cv::Mat resized_image;
    cv::resize(output_image, resized_image, cv::Size(resized_width, resized_height));
    imshow("Captured frame", resized_image);
}

cv::VideoCapture GetVideoCapture(const std::string& video_path, int image_width, int image_height) {
    if (video_path.empty()) {
        return cv::VideoCapture(
            GetGStreamerPipeline(/*capture_width=*/image_width, /*capture_height=*/image_height,
             /*display_width=*/image_width, /*display_height=*/image_height, /*frame_rate=*/10), cv::CAP_GSTREAMER);
    }
    return cv::VideoCapture(video_path);
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
    // const auto& vertices = first_sku_info.GetObjectVertices();
    // for (const auto& vertice : vertices) {
    //     std::cout << vertice.x << " " << vertice.y << " " << vertice.z << std::endl;
    // }
    
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

}  // namespace

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    std::string video_path = "";
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (!FLAGS_video_path.empty()) {
        // If a video path is given as an input, it processes the video.
        video_path = FLAGS_video_path;
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
    
    // Load the model information.
    std::unordered_map<std::string, perception::SKUInfo> model_info = IntializeModelInfo();
   // model_info["sku 1234567890"] = forklift;

    cv::VideoCapture capture = GetVideoCapture(video_path, FLAGS_image_width, FLAGS_image_height);
    const int kInitialFocusValue = 130;
    if (!capture.isOpened()) {
        LOG(ERROR) << "Could not open camera.";
        exit(EXIT_FAILURE);
    }
    cv::namedWindow("Captured frame");
    
    camera::Focuser focuser(/*bus_info=*/7, kInitialFocusValue);

    // Capture an OpenCV frame
    cv::Mat frame;
    std::vector<std::pair<std::string, std::vector<cv::Point2f>>> qr_results;

    auto qr_recognizer =  std::make_unique<perception::QrCodeRecognizer>(FLAGS_enable_debugging);
    if (FLAGS_enable_tracking) {
        qr_recognizer = std::make_unique<perception::MultiFrameBasedQrCodeRecognizer>(FLAGS_enable_debugging);
    }
    while (true) {
        capture >> frame;
        if (frame.empty()) {
            break;
        }
        // if (video_path.empty()) {
        //     // Run auto focus when the input is from the camera.
        //     auto focused = focuser.Focusing(original_frame); //, focus_roi);
        //     if (!focused) {
        //         continue;
        //     }
        // }
        cv::Mat output_image = frame.clone();
        qr_recognizer->DetectAndRecognize(frame, &qr_results, &output_image);
       
        if (FLAGS_enable_3d_box_fitting) {
            box_fitting->Run3DBoxFitting(frame, qr_results, qr_model, forklift, model_info, &output_image);
        }

        // Show captured frame, now with overlays!
        DisplayOutput(output_image, 0.5f);
                                                                                                                                                          
        int key = cv::waitKey(100) & 0xff;
            
        // Stop the program on the ESC key.
        if (key == 27) {
            break;
        }
    }
    capture.release();
    return 0;
}