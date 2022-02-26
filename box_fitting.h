#ifndef PERCEPTION_BOX_FITTING_H
#define PERCEPTION_BOX_FITTING_H

#include <assert.h>
#include <set>
#include <vector>
#include <unordered_map>

#include "opencv2/core/core.hpp"

namespace perception {

class ModelInfo {
 public:
  ModelInfo() {}
  ModelInfo(float width_mm, float height_mm, float depth_mm) 
    : width_mm_(width_mm), height_mm_(height_mm), depth_mm_(depth_mm) {
    Create3DVertices(width_mm_, height_mm_, depth_mm_);
  }
  ModelInfo(float width_mm, float height_mm, float depth_mm,
   float rotation_vector[3], float translation_vector[3]) 
    : width_mm_(width_mm), height_mm_(height_mm), depth_mm_(depth_mm) {
    Create3DVertices(width_mm_, height_mm_, depth_mm_);
    rotation_vector_ = cv::Mat(3, 1, cv::DataType<float>::type, rotation_vector);
    translation_vector_ = cv::Mat(3, 1, cv::DataType<float>::type, translation_vector);
  }

  // Gets the mutable feature variable.
  std::vector<cv::Point3f>* GetMutableFeatureVertices() {
    return &feature_vertices_;
  }

  const std::vector<cv::Point3f>& GetObjectVertices() const {
    return object_vertices_;
  }
  const std::vector<cv::Point3f>& GetFeatureVertices() const {
    return feature_vertices_;
  }

  const cv::Mat& GetRotationVector() const {
    return rotation_vector_;
  }

  const cv::Mat& GetTranslationVector() const {
    return translation_vector_;
  }

  float GetWidth() const { return width_mm_; }
  float GetHeight() const { return height_mm_; }
  float GetDepth() const { return depth_mm_; }
 
 private:
  void Create3DVertices(float width_mm, float height_mm, float depth_mm) {
    assert(width_mm_ >= 0 && height_mm_ >= 0 && depth_mm_ >= 0);
    // Generate a set of 3D vertices of the box model.
        //   5---6
    // 4-|-7 |
    // | 1-|-2
    // 0---3   
    object_vertices_ = {
      {0.f, 0.f, 0.f},
      {0.f, depth_mm_, 0.f},
      {width_mm_, depth_mm_, 0.f},
      {width_mm_, 0.f, 0.f},
      {0.f, 0.f, height_mm_},
      {0.f, depth_mm_, height_mm_},
      {width_mm_, depth_mm_, height_mm_},
      {width_mm_, 0.f, height_mm_}
    };
  }
  std::vector<cv::Point3f> object_vertices_;
  std::vector<cv::Point3f> feature_vertices_;
  float width_mm_;
  float height_mm_;
  float depth_mm_;
  cv::Mat rotation_vector_;
  cv::Mat translation_vector_;
};

class SKUInfo : public ModelInfo {
 public:
  SKUInfo() {}
  SKUInfo(const std::string& id, float width_mm, float height_mm, float depth_mm)
   : ModelInfo(width_mm, height_mm, depth_mm) {
    sku_id_ = id;
  }
  void SetWeight(float weight_kg) {
    weight_kg_ = weight_kg;
  }
  const std::string& GetSkuId() const { return sku_id_; }
  float GetWeight() const { return weight_kg_; }

 private:
  float weight_kg_;
  std::string sku_id_;
};

struct MatchedModelInfo {
    std::string sku_id;
    cv::Mat rvect;
    cv::Mat tvect;
};

class BoxFitting {
 public:
  BoxFitting(const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients);

  void Run3DBoxFitting(const cv::Mat& input,
      const std::vector<std::pair<std::string, std::vector<cv::Point2f>>>& qr_code_list, 
      const ModelInfo& qr_model,
      const ModelInfo& forklift,
      const std::unordered_map<std::string, SKUInfo>& model_info,
      cv::Mat* output_image);

 private:
  void RunPnP(
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point3f>& object_points,
     cv::Mat* rotation_vector, cv::Mat* translation_vector);
  std::vector<cv::Point2f> GetProjectedPoints(
      const std::vector<cv::Point3f>& point3d,
      const cv::Mat& rotation_vector, const cv::Mat& translation_vector);
  bool SearchBoxArea(const cv::Mat& frame, const perception::SKUInfo& box_info, const std::vector<cv::Point2f>& image_points,
    const perception::ModelInfo& qrcode_model, const cv::Mat& qrcode_rvec, const cv::Mat& qrcode_tvec,
    const perception::ModelInfo& forklift);
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
};

}  // namespace perception

#endif