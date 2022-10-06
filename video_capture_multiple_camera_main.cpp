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

std::string GetGStreamerPipeline(int camera_index) {
    return "v4l2src device=/dev/video" + std::to_string(camera_index) + " ! video/x-raw, format=UYVY, width=1920, height=1080 ! videoconvert ! video/x-raw, format=BGR ! appsink";
}

// std::string GetGStreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method=0) {
//     return std::string("v4l2src device=/dev/video4 ! video/x-raw, format=UYVY, width=1920, height=1080 ! videoconvert ! video/x-raw, format=BGR ! appsink");

constexpr int kImageWidth = 1920;
constexpr int kImageHeight = 1080;
constexpr int kFramerate = 10;

int SingleCameraOperation(int cam_idx, const string& video_file_name) {
    VideoCapture capture(GetGStreamerPipeline(cam_idx), cv::CAP_GSTREAMER);
    if (!capture.isOpened()) {
        cerr << "Could not open camera." << endl;
        return -1;
    }
    namedWindow("Captured frame");
    
    cv::Mat frame;
    capture >> frame;

    cerr << "Image size " << frame.size() << endl;
    // const int kCodec = VideoWriter::fourcc('M', 'P', '4', 'V');
    // string filename = "11.mp4";             // name of the output video file
    // writer.open(filename, kCodec, static_cast<double>(kFramerate), frame.size(), true);

    auto gstream_enc = std::string("appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw, format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420, width=1920, height=1080 ! omxh264enc  qp-range=20,20:20,20:-1,-1 ! matroskamux ! queue ! filesink location="+video_file_name+".mkv");
    VideoWriter writer = VideoWriter(gstream_enc, cv::CAP_GSTREAMER, 0, 15, {1920,1080});

    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    // Capture an OpenCV frame
    while (true) {
        auto t1 = std::chrono::steady_clock::now();
        capture >> frame;
        auto t2 = std::chrono::steady_clock::now();
       
        if (frame.empty()) {
            break;
        }
        imshow("Captured frame", frame);
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

int MultiCameraOperation(int num_cameras) {
    std::vector<VideoCapture> captures(num_cameras);
    std::vector<VideoWriter> writers(num_cameras);
    std::vector<string> window_names(num_cameras);

    cerr << "# of cameras " << num_cameras;
    for (int i = 0; i < num_cameras; ++i) {
        auto& capture = captures[i];
        auto& writer = writers[i];
        capture.open(GetGStreamerPipeline(i), cv::CAP_GSTREAMER);
        if (!capture.isOpened()) {
            cerr << "Could not open camera." << endl;
            return -1;
        }
        window_names[i] = std::to_string(i);
        namedWindow(window_names[i]);
        
        string video_file_name = "video_output" + window_names[i];
        auto gstream_enc = std::string("appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw, format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420, width=1920, height=1080 ! omxh264enc  qp-range=20,20:20,20:-1,-1 ! matroskamux ! queue ! filesink location=") + video_file_name + ".mkv";
        writer.open(gstream_enc, cv::CAP_GSTREAMER, 0, 15, {1920,1080});

        // check if we succeeded
        if (!writer.isOpened()) {
            cerr << "Could not open the output video file for write\n";
            return -1;
        }
    }

    // Capture an OpenCV frame
    while (true) {
        for (int i = 0; i < num_cameras; ++i) {
            auto& capture = captures[i];
            auto& writer = writers[i];
            cv::Mat frame;
            auto t1 = std::chrono::steady_clock::now();
            capture >> frame;
            auto t2 = std::chrono::steady_clock::now();
        
            if (frame.empty()) {
                break;
            }
            imshow(window_names[i], frame);
            writer.write(frame);
            auto t3 = std::chrono::steady_clock::now();
            cout << "capturing in ms:" << to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) << endl;
            cout << "writing in ms:" << to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) << endl;
        }

        int key = waitKey(1) & 0xff;
        // Stop the program on the ESC key.
        if (key == 27) {
            break;
        }
    }
    for (int i = 0; i < num_cameras; ++i) {
        captures[i].release();
        cout << "writing is done" << endl;
        writers[i].release();
    }
    return 0;
}

}  // namespace

// Image capture
int main(int argc, char **argv) {
    if (argc == 3) {
        int cam_idx = atoi(argv[1]);
        string video_file_name = argv[2];
        cerr << "Camera index " << cam_idx << " " << video_file_name << endl;

        return SingleCameraOperation(cam_idx, video_file_name);
    } else if (argc == 2) {
        int num_cameras = atoi(argv[1]);
        cerr << "# of cameras " << num_cameras << endl;
        return MultiCameraOperation(num_cameras);
    }
    return 0;
}