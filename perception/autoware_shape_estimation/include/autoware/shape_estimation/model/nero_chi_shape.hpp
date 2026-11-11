// Copyright 2026 Nero (wallbox_1107). Licensed under the Apache License, Version 2.0.

#ifndef AUTOWARE__SHAPE_ESTIMATION__MODEL__NERO_CHI_SHAPE_HPP_
#define AUTOWARE__SHAPE_ESTIMATION__MODEL__NERO_CHI_SHAPE_HPP_

#include <autoware_perception_msgs/msg/shape.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <array>
#include <vector>

namespace autoware::shape_estimation
{
namespace model
{
// NERO_SHAPE_UNKNOWN_CHI (wallbox_1107): is the switch set
bool neroChiEnabled();
// field 7 (wallbox-4): the chi-shape only for clusters whose hull comes within <corridor> m of the planned trajectory
bool neroChiCorridorWanted();
// the planned trajectory in the clusters' frame for the message being processed (nullptr = none)
void neroChiSetCorridor(const std::vector<std::array<double, 2>> * path_in_cluster_frame);
// replace a convex-hull footprint (POLYGON, relative to <pose>) by the chi-shape of the cluster's points; false = unchanged
bool neroChiFootprint(
  const pcl::PointCloud<pcl::PointXYZ> & cluster, const geometry_msgs::msg::Pose & pose,
  autoware_perception_msgs::msg::Shape & shape);
}  // namespace model
}  // namespace autoware::shape_estimation

#endif  // AUTOWARE__SHAPE_ESTIMATION__MODEL__NERO_CHI_SHAPE_HPP_
