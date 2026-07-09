# Launching modes
## ClipSeg Navigation based purely on vision (no obstacle avoidance)
```bash
# 1. Start RealSense with aligned depth
ros2 launch sensors_bringup realsense.launch.py 

# 2. Publish static sensor transforms
ros2 run tf2_ros static_transform_publisher \
    0.125 -0.15 0.825 0 0 0 \
    panther/base_link camera_link
ros2 run tf2_ros static_transform_publisher \
    0 0 0 -1.57079632679 0 -1.57079632679 \
    camera_link camera_depth_optical_frame

# 3. Run the object-following node
ros2 run robot_navigation clipseg_depth_node_navigation \
    --ros-args  -p rgb_topic:=/camera0/driver/color/image_raw \
    -p depth_topic:=/camera0/driver/aligned_depth_to_color/image_raw \
    -p "prompts:=['person']"
```

## Agro-GPT Final launch - CLIPSeg Navigation with Nav2
After launching the docker, move into the workspace:
```bash
cd /rgbd_camera_intel_dev
```
Use the full launch sequence:
```bash
ros2 launch robot_navigation target_pipeline.launch.py target_prompt:=person
```

or individual steps:
```bash
# 1. Start LiDAR
ros2 launch sensors_bringup velodyne.launch.py 

# 2. Start RealSense with aligned depth
ros2 launch sensors_bringup realsense.launch.py 

# 3. Publish static sensor transforms
# ros2 run tf2_ros static_transform_publisher \
#     0.125 -0.15 0.825 0 0 0 \
#     panther/base_link camera_link
# ros2 run tf2_ros static_transform_publisher \
#     0 0 0 -1.57079632679 0 -1.57079632679 \
#     camera_link camera_depth_optical_frame
# ros2 run tf2_ros static_transform_publisher \
#     0.125 0.02 0.825 0 0 0 \
#     panther/base_link panther/velodyne_link

ros2 run tf2_ros static_transform_publisher \
    0.125 -0.15 0.643 0 0 0 \
    panther/base_link camera_link
ros2 run tf2_ros static_transform_publisher \
    0 0 0 -1.57079632679 0 -1.57079632679 \
    camera_link camera_depth_optical_frame
ros2 run tf2_ros static_transform_publisher \
    0.125 0.02 0.643 0 0 0 \
    panther/base_link panther/velodyne_link

# 4. Start CLIPSeg using aligned depth
ros2 run robot_navigation clipseg_depth_node_with_mask \
    --ros-args \
    -p rgb_topic:=/camera0/driver/color/image_raw \
    -p depth_topic:=/camera0/driver/aligned_depth_to_color/image_raw \
    -p "prompts:=['person']"

# 5. Publish /target_point
ros2 run robot_navigation masked_depth_target_point \
    --ros-args \
    -p masked_depth_topic:=/clipseg/masked_depth \
    -p camera_info_topic:=/camera0/driver/aligned_depth_to_color/camera_info \
    -p target_point_topic:=/target_point

# 6. Generate target ROI point cloud from masked depth
ros2 run robot_navigation rgbd2pointcloud

# 7. Subtract target ROI from LiDAR cloud
ros2 run robot_navigation crop_lidar

  # 8. Map -> odom static transform
# Temporary map->odom transform, for testing only
# ros2 run tf2_ros static_transform_publisher \
#   0 0 0 0 0 0 \
#   map panther/odom
# For actual SLAM
ros2 launch robot_navigation slam.launch.py

# 9. Start Nav2
ros2 launch robot_navigation target_follow.launch.py
```


## New features
Launching pipeline with compressed camera feed and compressed /clipseg/debug_image visualization
```bash
ros2 launch robot_navigation target_pipeline_compressed.launch.py target_prompt:=person
# For full laptop
ros2 launch robot_navigation target_pipeline_compressed_full.launch.py target_prompt:=person
ros2 launch robot_navigation target_pipeline_compressed_nvblox.launch.py target_prompt:=person

ros2 launch robot_navigation panther_rviz.launch.py 
```

## Frontier exploration
```bash
# Launch pipeline
ros2 launch robot_navigation panther_frontier_exploration.launch.py 
# Go to next frontier goal
ros2 service call /panther_frontier_nav2_client/step_once std_srvs/srv/Trigger {}
# Start continous exploration
ros2 service call /panther_frontier_nav2_client/start std_srvs/srv/Trigger {}
# Stop
ros2 service call /panther_frontier_nav2_client/stop std_srvs/srv/Trigger {}

# Even cleaner
ros2 launch robot_navigation panther_frontier_exploration.launch.py \
  frontier_marker_max_count:=15 \
  frontier_marker_scale:=0.05 \
  frontier_region_size_thresh:=20

ros2 bag record -o /media/vladkobli/SSDPortable/rosbags/ /panther/odometry/filtered /panther/imu/data /velodyne_points /marker /map /pose