// Copyright 2026 Nero
// [nero camstack 2026-11-10] roi_class_filter in C++ (replaces autoware_nero_camera0's Python roi_class_filter on the car: a Python node at
// 10 Hz cost ~0.12-0.17 NeroStation cores in the replays; same parameters, topics and output): the camera boxes allowed to LABEL lidar
// clusters (roi_cluster_fusion): only the classes in keep_classes (PEDESTRIAN by default), score >= min_score, not within border_px of the
// image border (boxes cut by the border are classified worst). The rest of the message is copied unchanged.
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace autoware::nero_camera0_rect
{
using tier4_perception_msgs::msg::DetectedObjectsWithFeature;

class RoiClassFilterNode : public rclcpp::Node
{
public:
  explicit RoiClassFilterNode(const rclcpp::NodeOptions & options) : rclcpp::Node("roi_class_filter", options)
  {
    static const std::map<std::string, uint8_t> labels = {
      {"UNKNOWN", 0}, {"CAR", 1}, {"TRUCK", 2}, {"BUS", 3}, {"TRAILER", 4}, {"MOTORCYCLE", 5}, {"BICYCLE", 6}, {"PEDESTRIAN", 7}};
    const auto keep = declare_parameter<std::vector<std::string>>("keep_classes", std::vector<std::string>{"PEDESTRIAN"});
    std::string ks;
    for (const auto & k : keep) {
      keep_.insert(labels.at(k));
      ks += (ks.empty() ? "" : ", ") + k;
    }
    min_score_ = declare_parameter<double>("min_score", 0.5);
    border_ = declare_parameter<int>("border_px", 4);
    w_ = declare_parameter<int>("image_width", 1440);
    h_ = declare_parameter<int>("image_height", 1080);
    const auto q = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
    pub_ = create_publisher<DetectedObjectsWithFeature>("~/output/rois", q);
    sub_ = create_subscription<DetectedObjectsWithFeature>(
      "~/input/rois", q, [this](DetectedObjectsWithFeature::ConstSharedPtr m) { on_rois(*m); });
    RCLCPP_INFO(
      get_logger(), "[NERO_CAMERA0] roi_class_filter (C++) keeps [%s], score >= %.2f, not within %d px of the %dx%d border", ks.c_str(),
      min_score_, border_, w_, h_);
  }

private:
  void on_rois(const DetectedObjectsWithFeature & m)
  {
    auto out = std::make_unique<DetectedObjectsWithFeature>();
    out->header = m.header;
    const int64_t b = border_, w = w_, h = h_;
    for (const auto & fo : m.feature_objects) {
      const auto & r = fo.feature.roi;
      if (fo.object.classification.empty() || !keep_.count(fo.object.classification.front().label)) continue;
      if (fo.object.existence_probability < min_score_) continue;
      if (!(int64_t(r.x_offset) > b && int64_t(r.y_offset) > b && int64_t(r.x_offset) + r.width < w - b &&
            int64_t(r.y_offset) + r.height < h - b)) {
        continue;
      }
      out->feature_objects.push_back(fo);
    }
    pub_->publish(std::move(out));
  }

  std::set<uint8_t> keep_;
  double min_score_{0.5};
  int border_{4}, w_{1440}, h_{1080};
  rclcpp::Publisher<DetectedObjectsWithFeature>::SharedPtr pub_;
  rclcpp::Subscription<DetectedObjectsWithFeature>::SharedPtr sub_;
};
}  // namespace autoware::nero_camera0_rect

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::nero_camera0_rect::RoiClassFilterNode)
