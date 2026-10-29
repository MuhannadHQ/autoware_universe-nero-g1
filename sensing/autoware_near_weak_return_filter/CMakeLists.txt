cmake_minimum_required(VERSION 3.14)
project(autoware_near_weak_return_filter)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/near_weak_return_filter.cpp
  src/near_weak_return_filter_node.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::near_weak_return_filter::NearWeakReturnFilterComponent"
  EXECUTABLE near_weak_return_filter_node
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_near_weak_return_filter test/test_near_weak_return_filter.cpp)
  target_link_libraries(test_near_weak_return_filter ${PROJECT_NAME})
endif()

ament_auto_package(INSTALL_TO_SHARE config)
