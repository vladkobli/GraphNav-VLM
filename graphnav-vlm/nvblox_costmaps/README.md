# nvblox_costmaps

ROS 2 package providing launch configurations for NvBlox-based costmap generation using RealSense depth cameras and/or Velodyne LiDAR sensors with a Husarion Panther UGV platform.

## Overview

This package contains launch files that start the NvBlox node along with the necessary sensor drivers and transform publishers for different sensor configurations. NvBlox provides GPU-accelerated 3D scene understanding and costmap generation for autonomous navigation.

## Supported Sensor Configurations

### 1. RealSense Only
**File:** `realsense_nvblox.launch.py`

Uses only an Intel RealSense depth camera for 3D mapping and costmap generation.

**Usage:**
```bash
ros2 launch nvblox_costmaps realsense_nvblox.launch.py
```

**Features:**
- RealSense depth camera input
- Static transform publisher (base_link → camera0_link)
- Single camera NvBlox configuration
- Optimized for real-time performance with a single depth sensor

---

### 2. Velodyne LiDAR Only
**File:** `velodyne_nvblox.launch.py`

Uses a Velodyne LiDAR for 3D mapping and costmap generation.

**Usage:**
```bash
ros2 launch nvblox_costmaps velodyne_nvblox.launch.py
```

**Features:**
- Velodyne driver initialization (VLP16 configuration)
- Point cloud generation from LiDAR packets
- Static transform publisher (base_link → velodyne_link)
- NvBlox configured for LiDAR input

---

### 3. RealSense + Velodyne
**File:** `realsense_velodyne_nvblox.launch.py`

Combines both RealSense depth camera and Velodyne LiDAR for enhanced 3D scene understanding.

**Usage:**
```bash
ros2 launch nvblox_costmaps realsense_velodyne_nvblox.launch.py
```

**Features:**
- Dual sensor fusion (RealSense + Velodyne)
- Both depth and LiDAR data processed simultaneously
- Best for comprehensive environmental mapping
- Improved robustness and coverage

---

### 4. RealSense + Velodyne (Autostart)
**File:** `autostart_realsense_velodyne_nvblox.launch.py`

Similar to the dual-sensor configuration but designed for automated startup scenarios.

**Usage:**
```bash
ros2 launch nvblox_costmaps autostart_realsense_velodyne_nvblox.launch.py
```

**Features:**
- Combines RealSense and Velodyne sensors
- Pre-configured for automatic startup on system boot
- Suitable for deployment and production scenarios

---

## Prerequisites

### System Requirements
- Ubuntu 22.04 or Ubuntu 24.04
- ROS 2 (Humble or newer)
- NVIDIA GPU (for NvBlox GPU acceleration)
- CUDA and cuDNN installed

### Required Packages
The following packages should be installed and built:
- `nvblox_ros` - Core NvBlox ROS 2 node
- `nvblox_ros_common` - Common utilities
- `nvblox_ros_python_utils` - Python utilities
- `isaac_ros_launch_utils` - Launch utilities
- `sensors_bringup` - Sensor driver configurations
- `velodyne_driver` - Velodyne LiDAR driver
- `velodyne_pointcloud` - Point cloud conversion
- `tf2_ros` - Transform broadcasting

### Hardware
- **RealSense Option:** Intel RealSense D455 or D435 depth camera
- **Velodyne Option:** Velodyne VLP16 LiDAR
- **Robot Platform:** Husarion Panther UGV (transforms configured for this platform)

---

## Configuration

### Transform Frames

All launch files define static transforms between the robot base and sensors:

**RealSense Transform:**
- Parent: `panther/base_link`
- Child: `camera0_link`
- Position (xyz): `[0.125, -0.15, 0.825]` meters
- Rotation (rpy): `[0, 0, 0]` radians

**Velodyne Transform:**
- Parent: `panther/base_link`
- Child: `panther/velodyne_link`
- Position (xyz): `[0.125, 0.020, 0.643]` meters
- Rotation (rpy): `[0, 0, 0]` radians

### NvBlox Parameters

Common NvBlox configuration:
- **Global Frame:** `panther/odom`
- **Pose Frame:** `panther/base_link`
- **Map Clearing Frame:** `panther/base_link`
- **Depth Input:** RealSense camera topics
- **LiDAR Input:** Velodyne point cloud topics
- **ESDF Slice Min Height:** 0.2 m (prevents ground detection as obstacle)

To customize parameters, edit the respective launch file and modify the NvBlox node's parameter dictionary.

---

## Running the Launches

### Step 1: Source Your Workspace
```bash
cd /workspaces/isaac_ros-dev
source install/setup.bash
```

### Step 2: Launch Your Configuration
Choose based on your available sensors:

```bash
# RealSense only
ros2 launch nvblox_costmaps realsense_nvblox.launch.py

# Velodyne only
ros2 launch nvblox_costmaps velodyne_nvblox.launch.py

# RealSense + Velodyne (combined)
ros2 launch nvblox_costmaps realsense_velodyne_nvblox.launch.py

# RealSense + Velodyne (autostart)
ros2 launch nvblox_costmaps autostart_realsense_velodyne_nvblox.launch.py
```

### Step 3: Verify in RViz
In another terminal, start RViz to visualize the costmap:
```bash
rviz2
```

Add displays for:
- **Esdf slice:** Shows the 2D occupancy grid at robot height
- **Point cloud:** Visualizes input sensor data
- **Transforms:** Verifies TF tree structure

---

## Troubleshooting

### Issue: "Failed to find sensor data"
- **RealSense:** Ensure the camera is connected and accessible
  ```bash
  rs-enumerate-devices
  ```
- **Velodyne:** Verify the LiDAR network connection (typically 192.168.1.201)

### Issue: "Transform lookup failed"
- Verify transform frames in RViz
- Check that sensor drivers are publishing transforms
- Ensure `use_sim_time` parameter is correctly set (False for hardware, True for simulation)

### Issue: NvBlox not publishing outputs
- Check that input topics match the configured sensor topics
- Verify GPU is available and CUDA drivers are installed
- Check NvBlox logs for memory or performance issues

### Issue: Slow performance or high latency
- Reduce NvBlox resolution parameters
- Lower the maximum range for depth cameras
- Disable unused input sources (depth or LiDAR)

---

## Development

### Adding a New Launch Configuration

1. Create a new `.launch.py` file in the `launch/` directory
2. Follow the existing launch file structure
3. Update [setup.py](setup.py) to include the new launch file in `data_files`
4. Update this README with documentation

### Testing Your Launch
```bash
# Check for Python syntax errors
python3 -m py_compile launch/my_new_launch.launch.py

# Validate ROS 2 launch structure
ros2 launch nvblox_costmaps my_new_launch.launch.py --show-args
```

---

## References

- **NvBlox GitHub:** https://github.com/nvidia-isaac/nvblox
- **ROS 2 Documentation:** https://docs.ros.org/
- **Intel RealSense:** https://dev.intelrealsense.com/
- **Velodyne LiDAR:** https://velodynelidar.com/
- **Husarion Panther:** https://husarion.com/

---

## License

Apache License 2.0

## Author

Vladislav Koblitskiy (vladkoblicica1@gmail.com)
