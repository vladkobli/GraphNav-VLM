from launch import LaunchDescription
from launch_ros.actions import Node
import ament_index_python
import os

def generate_launch_description():
    share_dir = ament_index_python.packages.get_package_share_directory('sensors_bringup')
    params_file = os.path.join(share_dir, 'config', 'realsense_config_nvblox.yaml')

    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='camera0',  # for nvblox
        name='driver',
        output='screen',
        parameters=[params_file],
        arguments=['--ros-args', '--log-level', 'error']
    )

    return LaunchDescription([realsense_node])