#include "focuser.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace camera {

using namespace cv;

// Search space for best focus value.
constexpr int kMaxValue = 600;
constexpr int kMinValue = 200;
constexpr int kMaxConsecutiveNumDecreasingFrames = 6;
constexpr int kFocusValueIncremental = 10;

Focuser::Focuser(int bus_info, int initial_focus_value) : m_i2c_bus(bus_info) {
    // Set the camera focus to the given focus value.
    Reset(initial_focus_value);
}

double Focuser::ComputeSharpness(const cv::Mat& frame) {
    cv::Mat gray, laplace;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplace, CV_16U);
    
  //  cv::imshow("gray", gray);
    // Compute the total sum of lapacian output for relative comparison.
    return cv::sum(laplace)[0];
}

bool Focuser::Focusing(const cv::Mat& frame, const cv::Rect& roi) {
    return Focusing(frame(roi));
}

void Focuser::Reset(int focus_value) {
    m_focused = false;
    m_last_sharpness = 0;
    m_decreasing_count = 0;
    m_max_sharpness = 0.0;
    m_current_focus_value = focus_value;
    m_best_focus_value = m_current_focus_value;
    m_current_focus_value = 120;
    set(m_current_focus_value);
}

bool Focuser::IsResetRequired() {
    auto current_ms = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_ms - last_focused_timestamp).count();
    if (duration_ms > 10000) {
        std::cout << "Reset!!! Duration ms " << duration_ms << std::endl;
        Reset(kMinValue);
        return true;
    }
    return false;
}

bool Focuser::StopSearching() {
    return m_decreasing_count == kMaxConsecutiveNumDecreasingFrames || m_current_focus_value >= kMaxValue;
}

bool Focuser::Focusing(const cv::Mat& frame) {
    return true;
    if (m_focused) {// && !IsResetRequired()) {
        return true;
    }
    if (StopSearching()) {
        // If the sharpness consecutively decreases for the certain number of
        // frames, set the focus with the best sharpness. 
        m_decreasing_count = 0;
        std::cout << "best m_best_focus_value:" << m_best_focus_value << std::endl;
        set(m_best_focus_value);
        m_focused = true;
        last_focused_timestamp = std::chrono::steady_clock::now();
    } else {
        const auto sharpness = ComputeSharpness(frame);
        std::cout <<"sharpness " << sharpness << " " << m_max_sharpness  << std::endl;
        // Find the maximum image sharpness.
        if (sharpness > m_max_sharpness) {
            m_best_focus_value = m_current_focus_value;
            m_max_sharpness = sharpness;
        }
        // If the image sharpness starts decreasing consecutively.
        if (sharpness < m_max_sharpness) {
            m_decreasing_count++;
        } else {
            // Reset.
            m_decreasing_count = 0;
        }
        if (m_decreasing_count <= kMaxConsecutiveNumDecreasingFrames) {
            m_last_sharpness = sharpness;
            m_current_focus_value += kFocusValueIncremental;
            std::cout << "m_current_focus_value " << m_current_focus_value << std::endl;
            set(m_current_focus_value);    
        }
    }   
    return m_focused;
}

void Focuser::set(int value) {
    std::cout << "input value: " << value << std::endl;
    // Make sure the value is within the boundary.
    value = std::min(value, kMaxValue);
    value = std::max(value, kMinValue);
    // These values are recommeded by Ardu. Do not change.
    value = (value << 4) & 0x3ff0;
    const auto data1 = (value >> 8) & 0x3f;
    const auto data2 = value & 0xf0;

    // "0x0C" is the chip i2c address.
    const auto command = "i2cset -y " + std::to_string(m_i2c_bus) + " 0x0C " + std::to_string(data1) + " " + std::to_string(data2);
    std::cout << " command " << command << std::endl;
    const auto status = system(command.c_str());
    if (status < 0) {
        // Print out the error when the status is not ok.
        std::cout << "Error: " << strerror(errno) << std::endl;
    }        
}

}  // namespace camera