#include "opencv2/core/core.hpp"

namespace camera {

class Focuser {
  public:
    Focuser(int bus_info, int initial_focus_value);
    bool Focusing(const cv::Mat& frame);
    bool Focusing(const cv::Mat& frame, const cv::Rect& roi);
    void Reset(int focus_value);

  private:
    void set(int value);
    bool IsResetRequired();
    bool StopSearching();
    double ComputeSharpness(const cv::Mat& frame);
    int m_i2c_bus;
    int m_current_focus_value;
    bool m_focused;
    int m_decreasing_count;
    int m_best_focus_value;
    double m_max_sharpness;
    double m_last_sharpness;
    std::chrono::time_point<std::chrono::steady_clock> last_focused_timestamp;
};

}  // namespace camera