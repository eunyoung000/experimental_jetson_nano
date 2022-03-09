#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/dnn/dnn.hpp>
#include "zbar.h"

using namespace std;
using namespace cv;
using namespace zbar;

namespace {

std::string GetGStreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method=0) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// Initialize the parameters
constexpr float confThreshold = 0.5; // Confidence threshold
constexpr float nmsThreshold = 0.4;  // Non-maximum suppression threshold
constexpr int inpWidth = 416;        // Width of network's input image
constexpr int inpHeight = 416;       // Height of network's input image

// std::string GetGStreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method=0) {
//     return "v4l2src device=/dev/video0 ! image/jpeg !jpegdec !video/x-raw, format=I420 ! videoconvert ! appsink";
// }

class YoloQrCodeDetector {
 public:
    YoloQrCodeDetector() {
        // Yolo detector initialization.
        // Load names of classes
        m_classes_files_ = "/home/motion2ai/Dev/qrcode_test/yolo_qr/qrcode.names";
        ifstream ifs(m_classes_files_.c_str());
        string line;
        while (getline(ifs, line)) {
            cout << "# " << line << endl;
            m_classes_.push_back(line);
        }

        // Give the configuration and weight files for the model
        String modelConfiguration = "/home/motion2ai/Dev/qrcode_test/yolo_qr/qrcode-yolov3-tiny.cfg";
        String modelWeights = "/home/motion2ai/Dev/qrcode_test/yolo_qr/qrcode-yolov3-tiny_last.weights";

        // Load the network
        m_net_ = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        m_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);    
    }
    void YoloQRcodeDetection(const Mat& frame, Mat* detection_output_image) {
        // Create a 4D blob from a frame.
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            
        //Sets the input to the network
        m_net_.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        m_net_.forward(outs, getOutputsNames(m_net_));
            
        // Remove the bounding boxes with low confidence
        postprocess(outs, detection_output_image);
            
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = m_net_.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(*detection_output_image, label, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
    }
 private:
    // Get the names of the output layers
    vector<String> getOutputsNames(const cv::dnn::Net& net) {
        static vector<String> names;
        if (names.empty()) {
            //Get the indices of the output layers, i.e. the layers with unconnected outputs
            vector<int> outLayers = net.getUnconnectedOutLayers();
            
            //get the names of all the layers in the network
            vector<String> layersNames = net.getLayerNames();
            
            // Get the names of the output layers in names
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i) {
                names[i] = layersNames[outLayers[i] - 1];
            }
        }
        return names;
    }

    // Draw the predicted bounding box
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat* detection_output_image) {
        //Draw a rectangle displaying the bounding box
        rectangle(*detection_output_image, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
        
        //Get the label for the class name and its confidence
        string label = format("%.2f", conf);
        if (!m_classes_.empty()) {
            CV_Assert(classId < (int)m_classes_.size());
            label = m_classes_[classId] + ":" + label;
        }
        
        //Display the label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(*detection_output_image, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
        putText(*detection_output_image, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0), 1);
    }
    // Remove the bounding boxes with low confidence using non-maxima suppression
    void postprocess(const vector<Mat>& outs, Mat* detection_output_image) {
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        const int frame_width = detection_output_image->cols;
        const int frame_height = detection_output_image->rows;
        
        for (size_t i = 0; i < outs.size(); ++i) {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame_width);
                    int centerY = (int)(data[1] * frame_height);
                    int width = (int)(data[2] * frame_width);
                    int height = (int)(data[3] * frame_height);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
        
        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        vector<int> indices;
        cv::dnn::dnn4_v20190621::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                    box.x + box.width, box.y + box.height, detection_output_image);
        }
    }
    // Darknet.
    cv::dnn::Net m_net_;
    // Class names.
    std::vector<string> m_classes_;
    string m_classes_files_;
};

int elapsed_time_printing_count = 0;
// Renders the computation time to the output image.
void VisualizeElapsedTime(const string& label, const auto& begin_ms, const auto& end_ms, cv::Mat* output) {
    auto elapsed_ms = label + " " + to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_ms - begin_ms).count());
    // Print out the computation time in ms at the top corner of the image.
    const int text_x = static_cast<int>(output->cols * 0.05f);
    const int text_y = static_cast<int>(output->rows * 0.05f);
    putText(*output, elapsed_ms, cv::Point(text_x, text_y + elapsed_time_printing_count * 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 5);
    elapsed_time_printing_count++;
}

class ZbarQrCodeRecognizer {
 public:
  ZbarQrCodeRecognizer() {
    // Configure the reader
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
  }
  // Renders the QR code results to the output image.
  // Note that the output image is in the original image coordinate system. 
  // If the result is from the cropped image, it translates the QR code recognition result by the roi to be aligned.
  void VisualizeOutput(const Image& image, const cv::Rect& roi, cv::Mat* output) {
    // Extract results
    for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
        putText(*output, symbol->get_data(), Point(symbol->get_location_x(0) + roi.x, symbol->get_location_y(0) + roi.y - 10), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);

        // Draw location of the symbols found
        if (symbol->get_location_size() == 4) {
            std::vector<cv::Point> points(4);
            for (int i = 0; i < 4; ++i) {
                points[i] = cv::Point(symbol->get_location_x(i) + roi.x, symbol->get_location_y(i) + roi.y);
            }
            cv::rectangle(*output, boundingRect(points), cv::Scalar(0, 255, 0), 10);
        }
    }
  }

  void RunRecognition(const cv::Mat& frame, cv::Rect roi, cv::Mat* output) {
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
    
    if (output != nullptr) {
        VisualizeElapsedTime("QR code computation in ms", begin_ms, end_ms, output);
        VisualizeOutput(image, roi, output);
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

std::vector<cv::Rect> GetRegionOfInterest(const Mat& frame) {
    float scale = 0.2f;
    const int resized_width = static_cast<int>(scale * frame.cols);
    const int resized_height = static_cast<int>(scale * frame.rows);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cout << resized_width << " "<<  resized_height << endl;
    cv::Mat resized_frame, resized_grayimage;
    cv::resize(frame, resized_frame, cv::Size(resized_width, resized_height));
    cv::cvtColor(resized_frame, resized_grayimage, cv::COLOR_BGR2GRAY);
    // Erode to fill the small gap in the qrcode pattern (black part).
    erode(resized_grayimage, resized_grayimage, element, Point(-1, -1), 2, 1, 1);
    // Delete the noise pattern.
    blur( resized_grayimage, resized_grayimage, Size(3,3) );
    cv::threshold(resized_grayimage, resized_grayimage, 50, 255, cv::THRESH_BINARY_INV);

    dilate(resized_grayimage, resized_grayimage, element, Point(-1, -1), 2, 1, 1);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(resized_grayimage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<cv::Rect> rois;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect bounding_box = boundingRect(contours[i]);
        vector<Point> contours_poly;
        float perimeter = arcLength(contours[i], true);
        approxPolyDP(contours[i], contours_poly, /*epsilon=*/0.04 * perimeter, /*closed=*/true);
        if (contours_poly.size() != 4) {
            continue;
        }
        float area = contourArea(contours[i]);
        float ratio = bounding_box.width / static_cast<float>(bounding_box.height);
        
        if (area < 200 || area > 600 || ratio < 0.8 || ratio > 1.3) {
            continue;
        }
        // Tranform the box back to the original coordinate system.
        ScaleBoundingBox(1.f / scale, &bounding_box);
        rois.push_back(bounding_box);
    }
    cout << contours.size() << " " << rois.size() << endl;
    return rois;
}

void DisplayOutput(const Mat& output_image, float scale) {
    const int resized_width = static_cast<int>(scale * output_image.cols);
    const int resized_height = static_cast<int>(scale * output_image.rows);
    cv::Mat resized_image;
    cv::resize(output_image, resized_image, Size(resized_width, resized_height));
    imshow("Captured frame", resized_image);
}

}  // namespace

int main(int argc, char **argv) {
    string video_path = "/home/motion2ai/Dev/videos/test_vides/full_res_95inches_6mm.mp4";

    if (argc == 2) {
        video_path = argv[1];
    }
    // VideoCapture capture(GetGStreamerPipeline(/*capture_width=*/1920, /*capture_height=*/1080,
    //  /*display_width=*/1920, /*display_height=*/1080, /*frame_rate=*/30), cv::CAP_GSTREAMER);
    VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        cerr << "Could not open camera." << endl;
        exit(EXIT_FAILURE);
    }
    namedWindow("Captured frame");
    
    // YoloQrCodeDetector yolo_qr_detector;
    ZbarQrCodeRecognizer zbar_qr_recognizer;
    
    // Capture an OpenCV frame
    cv::Mat frame, frame_grayscale, original_frame;
    while (true) {
        capture >> original_frame;
        if (original_frame.empty()) {
            break;
        }
        elapsed_time_printing_count = 0;
        auto begin_ms = std::chrono::steady_clock::now();
        const auto& rois = GetRegionOfInterest(original_frame);
        auto end_ms = std::chrono::steady_clock::now();
        if (rois.empty()) {
            continue;
        }
        cv::Mat output_image = original_frame.clone();
        VisualizeElapsedTime("ROI computation in ms", begin_ms, end_ms, &output_image);
        for (auto& roi : rois) {
            const auto cropped = original_frame(roi);
            // imwrite("test.jpg", frame);

            cout << "roi resolution: " << cropped.cols << " " << cropped.rows << endl;
            zbar_qr_recognizer.RunRecognition(cropped, roi, &output_image);
            cv::rectangle(output_image, roi, cv::Scalar(255, 0, 0), 10);
        }

        //Show captured frame, now with overlays!
        DisplayOutput(output_image, 0.5f);
                                                                                                                                                          
        int key = waitKey(1) & 0xff;
            
        // Stop the program on the ESC key.
        if (key == 27) {
            break;
        }
    }
    capture.release();
    return 0;
}