#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>

#include "focuser.h"
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

}  // namespace

// Image capture
int main(int argc, char **argv) {
    int cam_idx = 0;

    if (argc == 2) {
        cam_idx = atoi(argv[1]);
    }
    VideoCapture capture(GetGStreamerPipeline(/*capture_width=*/4032, /*capture_height=*/3040,
    /*display_width=*/4032, /*display_height=*/3040, /*frame_rate=*/10), cv::CAP_GSTREAMER);
    if (!capture.isOpened()) {
        cerr << "Could not open camera." << endl;
        exit(EXIT_FAILURE);
    }
    namedWindow("Captured frame");
    
    camera::Focuser focuser(/*bus_info=*/7, /*initial_focus_value=*/120);

    cv::Mat frame;
    capture >> frame;
    VideoWriter writer;
    int codec = VideoWriter::fourcc('M', 'P', '4', 'V');
    double fps = 1.0;                          // framerate of the created video stream
    string filename = "/home/motion2ai/Dev/videos/test_vides/fixed_focus_qr_test.mp4";             // name of the output video file
    writer.open(filename, codec, fps, frame.size(), true);
    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    int skipped_frame_count = 0;
    // Capture an OpenCV frame
    while (true) {
        auto t1 = std::chrono::steady_clock::now();
        capture >> frame;
        auto t2 = std::chrono::steady_clock::now();
       
        if (frame.empty()) {
            break;
        }
        cv::Mat frame_resized;
        resize(frame, frame_resized, Size(1920, 1080));
        
        // imshow("Captured frame", frame_resized);

        auto focused = focuser.Focusing(frame);
        if (!focused) {
            continue;
        }
        // writer.write(frame);
        auto t3 = std::chrono::steady_clock::now();
        cout << "capturing in ms:" << to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) << endl;
        cout << "writing in ms:" << to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) << endl;

        int key = waitKey(1) & 0xff;
        // Stop the program on the ESC key.
        if (key == 27) {
            break;
        }
    }
    capture.release();
    cout << "writing is done" << endl;
    writer.release();
    return 0;
}