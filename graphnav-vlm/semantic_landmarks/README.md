# Semantic Landmark Logging

## Launching Realsense
```bash
# Launching RealSense camera
ros2 launch sensors_bringup realsense.launch.py 

# Launch compressed topic publishing
ros2 launch robot_navigation downsampled_realsense.launch.py
```

## Launching Landmark Mapping
```bash
ros2 launch semantic_landmarks target_pipeline_landmarks.launch.py target_prompt:="person,couch,fan,shirt"

ros2 launch semantic_landmarks landmark_mapping.launch.py target_prompt:="person,couch,fan,shirt"

ros2 launch semantic_landmarks landmark_mapping_2.launch.py target_prompt:="person,couch,fan,shirt"
```