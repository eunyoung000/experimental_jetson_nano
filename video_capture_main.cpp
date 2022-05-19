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
    return "nvarguscamerasrc aelock=true gainrange=\"7 7\" exposuretimerange=\"5000000 5000000\" !  video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

constexpr int kImageWidth = 1920;
constexpr int kImageHeight = 1080;
constexpr int kFramerate = 10;

}  // namespace

// Image capture
int main(int argc, char **argv) {
    int cam_idx = 0;

    if (argc == 2) {
        cam_idx = atoi(argv[1]);
    }
    VideoCapture capture(GetGStreamerPipeline(/*capture_width=*/kImageWidth, /*capture_height=*/kImageHeight,
    /*display_width=*/kImageWidth, /*display_height=*/kImageHeight, /*frame_rate=*/kFramerate), cv::CAP_GSTREAMER);
    if (!capture.isOpened()) {
        cerr << "Could not open camera." << endl;
        exit(EXIT_FAILURE);
    }
    namedWindow("Captured frame");
    
    camera::Focuser focuser(/*bus_info=*/7, /*initial_focus_value=*/120);

    cv::Mat frame;
    capture >> frame;
    VideoWriter writer;
    const int kCodec = VideoWriter::fourcc('M', 'P', '4', 'V');
    string filename = "../videos/11.mp4";             // name of the output video file
    writer.open(filename, kCodec, static_cast<double>(kFramerate), frame.size(), true);
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
        imshow("Captured frame", frame);

        // auto focused = focuser.Focusing(frame);
        // if (!focused) {
        //     continue;
        // }
        writer.write(frame);
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