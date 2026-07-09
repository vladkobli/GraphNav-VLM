# Sensors_bringup
## Launching camera/LiDAR/GPS
You can launch each sensor with the following commands:
```
# Launching all the sensors: camera, LiDAR and RTK GPS
ros2 launch sensors_bringup full_sensor_bringup.launch.py

# Launching the full bringup
ros2 launch sensors_bringup full_bringup.launch.py

# Launching only the RealSense camera
ros2 launch sensors_bringup realsense.launch.py

# Launching only the Velodyne LiDAR
ros2 launch sensors_bringup velodyne.launch.py

# Launching only the RTK GPS
ros2 launch sensors_bringup gps.launch.py
```
