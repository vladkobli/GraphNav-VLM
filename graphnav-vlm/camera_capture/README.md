# Camera Capture

## Using GUI (Click to capture)
This mode uses a GUI with a capture button that records and saves all the requested data when pressed
```bash
# Launch
ros2 run camera_capture camera_capture --ros-args \
  -p realsense_color_topic:=/camera/driver/color/image_raw \
  -p realsense_aligned_depth_topic:=/camera/driver/aligned_depth_to_color/image_raw \
  -p gmsl_cam0_topic:=/camera0/image_color \
  -p gmsl_cam1_topic:=/camera1/image_color \
  -p gmsl_cam2_topic:=/camera2/image_color \
  -p gmsl_cam3_topic:=/camera3/image_color \
  -p save_dir:=/rgbd_camera_intel_dev/realsense_captures \
  -p require_all_topics:=false
```

## Automatic Waypoint Following
An RViz2 windows pops up, using SLAM. Explore the environment a bit so that the map is generated, and you can select the desired waypoints accordingly. Another option is to use a presaved map and use the the Estimate Pose option.
```bash
# Launch
ros2 launch camera_capture dataset_logging.launch.py \
    rviz_config:="/rgbd_camera_intel_dev/src/camera_capture/rviz/dataset_logging2.rviz"

# Trigger Waypoint following
ros2 service call /waypoint_logging/start std_srvs/srv/Trigger {}

# Clear current Waypoints
ros2 service call /waypoint_logging/clear std_srvs/srv/Trigger
```

## Semiautomatic dataset acuqisition
This test-case assumes the user teleoperates the robot in the desired interest loccations and presses the capture button on the GUI, which triggers the automatic acquisition sequence: photos in quadrature from 3 different poses: at 0 deg, 30 deg and 60 deg relative yaw. The pose of the robot is recorded for each node as well.
```bash
ros2 launch camera_capture manual_pose_sequence_logging.launch.py use_rviz:=true
```

## Lerobot
```bash
# Realsense launch
ros2 launch sensors_bringup realsense.launch.py
# Lerobot bringup old
ros2 launch camera_capture lerobot_realsense_sequence_logging.launch.py \
  use_lidar:=true \
  use_slam:=true \
  use_rviz:=true \
  lerobot_server_url:=http://172.17.0.1:8765

# Lerobot bringup new
ros2 launch camera_capture lerobot_realsense_sequence_logging.launch.py \
  use_lidar:=false \
  use_slam:=false \
  use_rviz:=false \
  lerobot_server_url:=http://172.17.0.1:8765
```

# Steps to build graph
1. Record dataset
2. Create map view with nodes
```bash
# Publish /map
ros2 bag play src/nodes/rosbag2_2026_06_15-14_11_24/rosbag2_2026_06_15-14_11_24_0.mcap --loop

# Publish markers
ros2 run camera_capture show_map_nodes \
  --ros-args \
  -p nodes_dir:=/rgbd_camera_intel_dev/src/nodes \
  -p map_topic:=/map \
  -p marker_topic:=/metadata_node_markers

# Generate high quality map
python3 src/camera_capture/camera_capture/build_high_quality_map.py --ros-args \
  -p map_topic:=/map \
  -p marker_topic:=/metadata_node_markers \
  -p output_prefix:=map_export \
  -p dpi:=600
```

3. Manually form connections between nodes to form graph and load them into `build_dataset_graph.py`
4. Final json graph
```bash
python3 src/camera_capture/camera_capture/build_dataset_graph.py \
  --dataset-root /rgbd_camera_intel_dev/src/nodes \
  --output-all8 /rgbd_camera_intel_dev/src/nodes_graph_8_color.json \
  --output-cardinal4 /rgbd_camera_intel_dev/src/nodes_graph_4_cardinal.json \
  --map-name indoor-outdoor \
  --fov-deg 70
```
