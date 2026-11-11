cmake_minimum_required(VERSION 3.14)
project(autoware_nero_camera0_rect)
# [nero camstack 2026-11-10] camera0 (the forward wide camera) C++ image path: replaces autoware_nero_camera0's Python camera0_node
find_package(autoware_cmake REQUIRED)
autoware_package()
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavutil libswscale)
ament_auto_add_library(${PROJECT_NAME} SHARED src/camera0_rect_node.cpp src/roi_class_filter_node.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} PkgConfig::LIBAV)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::nero_camera0_rect::Camera0RectNode"
  EXECUTABLE camera0_rect_node)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::nero_camera0_rect::RoiClassFilterNode"
  EXECUTABLE roi_class_filter_node)
ament_auto_package(INSTALL_TO_SHARE launch config)
