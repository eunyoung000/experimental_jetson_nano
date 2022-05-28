#include "box_fitting.h"

#include <iostream>
#include <sstream>

#include "glog/logging.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace perception {
namespace {

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


void RunCornerDetection(const cv::Mat& input) {
    cv::Mat resized_frame;
    cv::Mat gray, dst;

    // cv::blur( resized_frame, resized_frame, Size(3,3) );
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate(input, resized_frame, element, cv::Point(-1, -1), /*iterations=*/2, /*borderType=*/1, 1);
    cv::erode(resized_frame, resized_frame, element, cv::Point(-1, -1),  /*iterations=*/3, /*borderType=*/1, 1);

    cv::cvtColor(resized_frame, gray, cv::COLOR_BGR2GRAY);
    imshow("corner", gray);
}

std::vector<cv::Point3f> GenerateProposals(const std::vector<cv::Point3f> &qr_points, const perception::SKUInfo& box_info, float height) {
    float search_stride = 10.f; // search for every 1 cm.
    float max_x = -1.0f;
    float max_y = -1.0f;
    float min_x = 10000.f;
    float min_y = 10000.f;
    for (const auto& point : qr_points) {
        if (point.x > max_x) {
            max_x = point.x;
        }
        if (point.x < min_x) min_x = point.x;
        if (point.y > max_y) {
            max_y = point.y;
        }
        if (point.y < min_y) min_y = point.y;
    }
    const float start_x = max_x - box_info.GetWidth();
    const float start_y = max_y - box_info.GetDepth();
    std::vector<cv::Point3f> translations;
    for (float x = start_x; x < min_x; x += search_stride) {
        for (float y = start_y; y < min_y; y += search_stride) {
            translations.push_back(cv::Point3f(x, y, height));
        }
    }   
    return translations;
}

std::vector<cv::Point3f> Create3DProposalBoxVertices(const cv::Point3f& start_point, float width_mm, float height_mm, float depth_mm) {
    std::vector<cv::Point3f> generated_points;
    generated_points = {
        {start_point.x, start_point.y, start_point.z},
        {start_point.x, start_point.y + depth_mm, start_point.z},
        {start_point.x + width_mm, start_point.y + depth_mm, start_point.z},
        {start_point.x + width_mm, start_point.y, start_point.z},
        {start_point.x, start_point.y, start_point.z - height_mm},
        {start_point.x, start_point.y + depth_mm, start_point.z - height_mm},
        {start_point.x + width_mm, start_point.y + depth_mm, start_point.z - height_mm},
        {start_point.x + width_mm, start_point.y, start_point.z - height_mm},
    };
    return generated_points;
}

void drawProjected3DBox(const std::vector<cv::Point2f>& projected_points, cv::Mat* output_image) {
    const cv::Scalar kProjectedColor(255, 255, 0);
    if (projected_points.size() == 8) {
        cv::line(*output_image, projected_points[0], projected_points[1], kProjectedColor, 3);
        cv::line(*output_image, projected_points[0], projected_points[3], kProjectedColor, 3);
        cv::line(*output_image, projected_points[0], projected_points[4], kProjectedColor, 3);
        cv::line(*output_image, projected_points[1], projected_points[2], kProjectedColor, 3);
        cv::line(*output_image, projected_points[1], projected_points[5], kProjectedColor, 3);
        cv::line(*output_image, projected_points[2], projected_points[3], kProjectedColor, 3);
        cv::line(*output_image, projected_points[2], projected_points[6], kProjectedColor, 3);
        cv::line(*output_image, projected_points[3], projected_points[7], kProjectedColor, 3);
        cv::line(*output_image, projected_points[4], projected_points[5], kProjectedColor, 3);
        cv::line(*output_image, projected_points[4], projected_points[7], kProjectedColor, 3);
        cv::line(*output_image, projected_points[5], projected_points[6], kProjectedColor, 3);
        cv::line(*output_image, projected_points[6], projected_points[7], kProjectedColor, 3);
    } else if (projected_points.size() == 4) {
        cv::line(*output_image, projected_points[0], projected_points[1], kProjectedColor, 3);
        cv::line(*output_image, projected_points[1], projected_points[2], kProjectedColor, 3);
        cv::line(*output_image, projected_points[2], projected_points[3], kProjectedColor, 3);  
        cv::line(*output_image, projected_points[0], projected_points[3], kProjectedColor, 3);     
    }
}

// Renders the QR code results to the output image.
// Note that the output image is in the original image coordinate system. 
// If the result is from the cropped image, it translates the QR code recognition result by the roi to be aligned.
void VisualizeQrOutput(const std::vector<std::pair<std::string, std::vector<cv::Point2f>>>& qr_code_list, cv::Mat* output) {
    // Extract results.
    for (const auto& qrcode : qr_code_list) {
        // Draw location of the symbols found
        cv::rectangle(*output, boundingRect(qrcode.second), cv::Scalar(0, 255, 0), 10);
    }
}

void drawModelInfo(const perception::SKUInfo& sku_info, const cv::Point2f& location, cv::Mat* output_image) {
    const std::string sku_id = "id: " + sku_info.GetSkuId();
    putText(*output_image, sku_id, cv::Point(location.x + 80, location.y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
    const std::string sku_cbm = "cbm: " + std::to_string(sku_info.GetWeight() * sku_info.GetHeight() * sku_info.GetDepth() / 1000000000.f);
    putText(*output_image, sku_cbm, cv::Point(location.x + 80, location.y + 60), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
    const std::string sku_weight = "weight (kg): " + std::to_string(sku_info.GetWeight());
    putText(*output_image, sku_weight, cv::Point(location.x + 80, location.y + 120), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
}


// Projects the 3D object in the object coorinate system onto the yz plane to create a side view.
cv::Point ConvertObjectCoordToImage(const cv::Point3f& point, int x_buffer, int y_buffer, int image_height) {
    return cv::Point(point.y + x_buffer, image_height - point.z + y_buffer);
}

void UpdateZoneAvailability(float width, float height, float depth, const std::vector<cv::Point3f>& object_area,
  std::vector<bool>* is_zone_available) {
    float half_width = width * 0.5f;
    float half_height = height * 0.5f;
    float half_depth = depth * 0.5f;
    int zone_id = -1;

    for (int i = 0; i < (int) object_area.size(); ++i) {
        
        zone_id = -1;
        if (object_area[i].x > half_width && object_area[i].x <= width && object_area[i].y > half_height && object_area[i].y <=  height && object_area[i].z > half_depth && object_area[i].z <=  depth) {
            zone_id = 7;
        } else if (object_area[i].x > half_width && object_area[i].x <= width && object_area[i].y > half_height && object_area[i].y <=  height && object_area[i].z >= 0 && object_area[i].z <=  half_depth) {
            zone_id = 6;
        } else if (object_area[i].x > half_width && object_area[i].x <= width && object_area[i].y >= 0 && object_area[i].y <=  half_height && object_area[i].z > half_depth && object_area[i].z <=  depth) {
            zone_id = 5;
        } else if (object_area[i].x > half_width && object_area[i].x <= width && object_area[i].y >= 0 && object_area[i].y <=  half_height && object_area[i].z >= 0 && object_area[i].z <=  half_depth) {
            zone_id = 4;
        } else if (object_area[i].x >= 0 && object_area[i].x <= half_width && object_area[i].y > half_height && object_area[i].y <=  height && object_area[i].z > half_depth && object_area[i].z <=  depth) {
            zone_id = 3;
        } else if (object_area[i].x >= 0 && object_area[i].x <= half_width && object_area[i].y > half_height && object_area[i].y <=  height && object_area[i].z >= 0 && object_area[i].z <=  half_depth) {
            zone_id = 2;
        } else if (object_area[i].x >= 0 && object_area[i].x <= half_width && object_area[i].y >= 0 && object_area[i].y <=  half_height && object_area[i].z > half_depth && object_area[i].z <=  depth) {
            zone_id = 1;
        } else if (object_area[i].x >= 0 && object_area[i].x <= half_width && object_area[i].y >= 0 && object_area[i].y <=  half_height && object_area[i].z >= 0 && object_area[i].z <=  half_depth) {
            zone_id = 0;
        } 
        std::cout << "object_area " << object_area[i] << " " << zone_id <<  std::endl;  
        (*is_zone_available)[zone_id] = zone_id != -1 ? true : false ;
    } 
}

void DisplayProjected3DViewFromSide(
    const perception::ModelInfo& forklift,
    const std::unordered_map<std::string, perception::SKUInfo>& model_info,
    const std::vector<MatchedModelInfo>& matched_sku_info) {
    const int kXBuffer = 200;
    const int kYBuffer = -100;
    // Object to camera coordinate
    cv::Mat object_2_camera;
    Rodrigues(forklift.GetRotationVector(), object_2_camera);

    std::vector<cv::Point3f> dstPoint;
    float scale = 0.2;
    cv::Mat projection_image(1000, 1000, CV_8UC3, cv::Scalar(0, 0, 100));

    const auto& forklift_vertices = forklift.GetObjectVertices();
    // Forklift projection: World coordinate system.
    std::vector<cv::Point2f> projected_points(forklift_vertices.size());
    for (int i = 0; i < (int) forklift_vertices.size(); ++i) {
        projected_points[i] = ConvertObjectCoordToImage(forklift_vertices[i] * scale, kXBuffer, kYBuffer, projection_image.rows);
    }

    // Convert the Camera coordinate system to the world.
    cv::Mat cameraVectorMat = (cv::Mat_<float>(3, 1) << 0.f, 0.f, 300.f);
    cv::Mat cameraUpVectorMat = (cv::Mat_<float>(3, 1) << 0.f, 300.f, 0.f);
    cv::Mat cameraOriginMat = (cv::Mat_<float>(3, 1) << 0.f, 0.f, 0.f);
    cv::Mat camera2object_rot = object_2_camera.inv();

    cv::Mat_<float> cameraVectorMatInObject = camera2object_rot * (cameraVectorMat - forklift.GetTranslationVector());
    cv::Mat_<float> cameraOriginMatInObject = camera2object_rot * (cameraOriginMat - forklift.GetTranslationVector());
    cv::Mat_<float> cameraUpVectorMatInObject = camera2object_rot * (cameraUpVectorMat - forklift.GetTranslationVector());
    cv::Point end1 = ConvertObjectCoordToImage(cv::Point3f(cameraOriginMatInObject) *scale, kXBuffer, kYBuffer, projection_image.rows);
    cv::Point end2 = ConvertObjectCoordToImage(cv::Point3f(cameraVectorMatInObject) * scale, kXBuffer, kYBuffer, projection_image.rows);
    cv::Point end3 = ConvertObjectCoordToImage(cv::Point3f(cameraUpVectorMatInObject) * scale, kXBuffer, kYBuffer, projection_image.rows);

    // std::cout << "cameraOriginMatInObject " << cameraOriginMatInObject << std::endl;
    // std::cout << "cameraVectorMatInObject " << cameraVectorMatInObject << std::endl;
 
    // Draw the forklift shape.
    cv::Point lift_top(end1.x, projected_points[5].y);
    cv::Point lift_bottom(end1.x, projected_points[2].y);
    cv::Point lift_right(projected_points[2].x + 50, projected_points[0].y);
    cv::line(projection_image, lift_bottom, lift_top, cv::Scalar(0, 0, 0), 7);
    cv::line(projection_image, lift_bottom, lift_right, cv::Scalar(0, 0, 0), 7);
    // Forklift loading area.
    cv::line(projection_image, projected_points[0], projected_points[1], cv::Scalar(0, 255, 0), 5);
    cv::line(projection_image, projected_points[1], projected_points[5], cv::Scalar(0, 255, 0), 5);
    cv::line(projection_image, projected_points[4], projected_points[5], cv::Scalar(0, 255, 0), 5);
    cv::line(projection_image, projected_points[4], projected_points[0], cv::Scalar(0, 255, 0), 5);
    // Camera
    cv::line(projection_image, end1, end2, cv::Scalar(255, 0, 0), 2);
    cv::line(projection_image, end1, end3, cv::Scalar(0, 255, 0), 2);
    cv::circle(projection_image, end1, 3, cv::Scalar(255, 255, 255), 5);

    cv::rectangle(projection_image, cv::Rect(10, 10, 980, 300), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(projection_image, cv::Rect(10, 10, 980, 300), cv::Scalar(0, 0, 0), 5, 4);

    std::cout << " matched_model size" << matched_sku_info.size() << std::endl;

    int potential_box_num = 0;
    float max_height_mm = 0;
    // Assume that there is only type of box on the pallet.
    float model_cbm = 0;
    std::set<std::string> detected_sku_ids;
    float total_weight = 0.f;
    float total_cbm = 0.f;
    // Convert the detected box info into the world.
    for (const auto& matched_sku : matched_sku_info) {
        const auto& matched_model = model_info.at(matched_sku.sku_id);
        detected_sku_ids.insert(matched_sku.sku_id);
        cv::Mat object_2_camera;
        Rodrigues(matched_sku.rvect, object_2_camera);
        const auto& matched_object_vertices = matched_model.GetObjectVertices();
        std::vector<cv::Point3f> transformed_points(matched_object_vertices.size());
        float max_z = 0.f;
        for (int i = 0; i < (int) matched_object_vertices.size(); i++) {
            const auto& point =  matched_object_vertices[i];
            cv::Mat object_point = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
            cv::Mat_<float> object_point_in_camera = object_2_camera * object_point + matched_sku.tvect;
            cv::Mat_<float> object_point_in_forklift = camera2object_rot * (object_point_in_camera - forklift.GetTranslationVector());
            transformed_points[i] = cv::Point3f(object_point_in_forklift);
            if (max_z < transformed_points[i].z) {
                max_z = transformed_points[i].z;
            }
            projected_points[i] = ConvertObjectCoordToImage(cv::Point3f(object_point_in_forklift) * scale, kXBuffer, kYBuffer, projection_image.rows);
            LOG(INFO) << projected_points[i].x << " " << projected_points[i].y;
        }
        std::cout << " matched_model size" << matched_sku.sku_id << std::endl;
        model_cbm = matched_model.GetWidth() * matched_model.GetHeight() * matched_model.GetDepth();
        // Due to the error in the measurements, we allow some tolerance.
        const float height_buffer = matched_model.GetHeight() * 0.5f;
        int possible_box_count = static_cast<int>((max_z + height_buffer) / matched_model.GetHeight());
        potential_box_num += possible_box_count;
        total_weight += matched_model.GetWeight() * possible_box_count;
        total_cbm += model_cbm * possible_box_count;
        drawProjected3DBox(projected_points, &projection_image);
        max_height_mm = std::max(max_height_mm, max_z);
    }

    const int kHeightGap = 40;
    if (matched_sku_info.size() > 0) {
        const std::string text1 = "number of SKUs identified: " + std::to_string(detected_sku_ids.size());
        putText(projection_image, text1 , cv::Point(20, kHeightGap), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        const std::string text = "number of boxes identified: " +  std::to_string(matched_sku_info.size());
        putText(projection_image, text , cv::Point(20, 2 * kHeightGap), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        const std::string text2 = "number of boxes potentially on the pallet: " + std::to_string(potential_box_num);
        putText(projection_image, text2 , cv::Point(20, 3 * kHeightGap), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        const std::string text3 = "max height of the boxes on the pallet (mm): " + to_string_with_precision(max_height_mm, 2);
        putText(projection_image, text3 , cv::Point(20, 4 * kHeightGap), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

        const float forklift_cbm = forklift.GetWidth() * forklift.GetHeight() * forklift.GetDepth();
        const float occupancy_percentage = static_cast<float>(total_cbm) / static_cast<float>(forklift_cbm);

        const std::string percentage_used = "% of occupied potentially: " + to_string_with_precision(occupancy_percentage * 100, 2);
        putText(projection_image, percentage_used, cv::Point(20, 5 * kHeightGap), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        const std::string weight_text = "total weight on the pallet: (kg) " + to_string_with_precision(total_weight, 2);
        putText(projection_image, weight_text, cv::Point(20, 6 * kHeightGap), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
    }
    cv::imshow("projection_image", projection_image);
}

}  // namespace

BoxFitting::BoxFitting(
  const cv::Mat& camera_matrix,
  const cv::Mat& distortion_coefficients)
     : camera_matrix_(camera_matrix),
      distortion_coefficients_(distortion_coefficients) {}

// void BoxFitting::SetForkliftGroundPlane(
//   const cv::Mat& rotation_vector, const cv::Mat& translation_vector) {

// }

void BoxFitting::RunPnP(
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point3f>& object_points,
     cv::Mat* rotation_vector, cv::Mat* translation_vector) {
  cv::solvePnP(object_points, image_points, 
  camera_matrix_, distortion_coefficients_, *rotation_vector, *translation_vector, false);
}

std::vector<cv::Point2f> BoxFitting::GetProjectedPoints(
    const std::vector<cv::Point3f>& point3d,
    const cv::Mat& rotation_vector, const cv::Mat& translation_vector) {
  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(point3d, rotation_vector, translation_vector,
   camera_matrix_, distortion_coefficients_, projected_points);

  return projected_points;
}

void BoxFitting::Run3DBoxFitting(const cv::Mat& input,
    const std::vector<std::pair<std::string, std::vector<cv::Point2f>>>& qr_code_list, 
    const ModelInfo& qr_model,
    const ModelInfo& forklift,
    const std::unordered_map<std::string, SKUInfo>& model_info,
    cv::Mat* output_image) {
    bool known_qrcode = true;
    std::vector<MatchedModelInfo> matched_sku_info;

    if (qr_code_list.size() > 0) {
        for (const auto& qrcode : qr_code_list) {
            if (model_info.count(qrcode.first) > 0) {
                auto& matched_model = model_info.at(qrcode.first);
                if (known_qrcode) {
                    MatchedModelInfo matched_model_info;
                    matched_model_info.rvect = cv::Mat(3, 1, cv::DataType<float>::type);
                    matched_model_info.tvect = cv::Mat(3, 1, cv::DataType<float>::type);
                    RunPnP(qrcode.second, matched_model.GetFeatureVertices(), &matched_model_info.rvect, &matched_model_info.tvect);

                    auto projected_points = GetProjectedPoints(matched_model.GetObjectVertices(), matched_model_info.rvect, matched_model_info.tvect);
                    // for (int q = 0; q < projected_points.size(); ++q) {
                    //     std::cout << "projected: " <<  projected_points[q].x << " " << projected_points[q].y  << std::endl;
                    // }
                    drawProjected3DBox(projected_points, output_image);
                    drawModelInfo(matched_model, qrcode.second[1], output_image);
                    VisualizeQrOutput(qr_code_list, output_image);
                    matched_model_info.sku_id = qrcode.first;
                    matched_sku_info.push_back(matched_model_info);
                    
                } else {
                    cv::Mat qrcode_rvec = cv::Mat(3, 1, cv::DataType<float>::type);
                    cv::Mat qrcode_tvec = cv::Mat(3, 1, cv::DataType<float>::type);

                    // Compute the qrcode model pose w.r.t. the camera cooridate system.
                    RunPnP(qrcode.second, qr_model.GetObjectVertices(), &qrcode_rvec, &qrcode_tvec);
                    SearchBoxArea(input, matched_model, qrcode.second, qr_model, qrcode_rvec, qrcode_tvec, forklift);
                    auto projected_points = GetProjectedPoints(qr_model.GetObjectVertices(), qrcode_rvec, qrcode_tvec);
                    drawProjected3DBox(projected_points, output_image);
                }
            }
        }
    }
    DisplayProjected3DViewFromSide(forklift, model_info, matched_sku_info);
}

bool BoxFitting::SearchBoxArea(const cv::Mat& frame, const perception::SKUInfo& box_info, const std::vector<cv::Point2f>& image_points,
    const perception::ModelInfo& qrcode_model, const cv::Mat& qrcode_rvec, const cv::Mat& qrcode_tvec,
    const perception::ModelInfo& forklift) {
     float scale = 0.1f;
    const int resized_width = static_cast<int>(scale * frame.cols);
    const int resized_height = static_cast<int>(scale * frame.rows);

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(resized_width, resized_height));
    RunCornerDetection(resized_frame);

    // Convert the detected box info into the world.
    cv::Mat forklift_2_camera;
    Rodrigues(forklift.GetRotationVector(), forklift_2_camera);
    cv::Mat camera2forklift_rot = forklift_2_camera.inv();

    cv::Mat qrcode_2_camera;
    Rodrigues(qrcode_rvec, qrcode_2_camera);
    const auto& qrcode_model_vertices = qrcode_model.GetObjectVertices();
    std::vector<cv::Point3f> transformed_points(qrcode_model_vertices.size());
    float average_height = 0.f;
    for (int i = 0; i < (int) qrcode_model_vertices.size(); i++) {
        const auto& point =  qrcode_model_vertices[i];
        cv::Mat object_point = (cv::Mat_<float>(3, 1) << point.x, point.y, point.z);
        cv::Mat_<float> object_point_in_camera = qrcode_2_camera * object_point + qrcode_tvec;
        cv::Mat_<float> object_point_in_forklift = camera2forklift_rot * (object_point_in_camera - forklift.GetTranslationVector());
        transformed_points[i] = cv::Point3f(object_point_in_forklift);
        std::cout << "distance from the forklift ground: " << transformed_points[i].z << std::endl;
        average_height += transformed_points[i].z;
    }
    average_height /= static_cast<float>(qrcode_model_vertices.size());
    const auto& generated_proposals = GenerateProposals(transformed_points, box_info, average_height);
    cv::Mat test_output = frame.clone();
    for (const auto& proposal : generated_proposals) {
        std::vector<cv::Point3f> vertices = Create3DProposalBoxVertices(proposal, box_info.GetWidth(), box_info.GetHeight(), box_info.GetDepth());
        const auto& projected_points = GetProjectedPoints(vertices, forklift.GetRotationVector(), forklift.GetTranslationVector());
        drawProjected3DBox(projected_points, &test_output);
    }

    cv::Mat test_output_resized;
    cv::resize(test_output, test_output_resized, cv::Size(resized_width, resized_height));
    cv::imshow("proposals", test_output_resized);
    return true;
}

}  // namespace perception