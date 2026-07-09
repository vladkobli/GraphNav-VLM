cmake_minimum_required(VERSION 3.8)
project(sensors_bringup)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)

# Install launch files
install(
  DIRECTORY launch
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python node(s)
install(PROGRAMS
  src/get_roi_distance_node.py
  src/roi_overlay_node.py
  src/roi_overlay_node_interactive.py
  src/clipseg_depth_node.py
  src/clipseg_image_depth_node.py
  src/clipseg_depth_node_multi_instance.py
  src/clipseg_image_depth_node_multi_instance.py
  src/go_to_object_node.py
  src/clipseg_depth_node_navigation.py
  src/target_to_nav2_once.py
  src/target_to_nav2_continuous.py
  src/rgbd2pointcloud.py
  src/clipseg_depth_node_with_mask.py
  src/clipseg_depth_node_river.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
