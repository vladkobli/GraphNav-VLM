import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # --- Locate each package's launch files ---
    realsense_launch = os.path.join(
        get_package_share_directory('sensors_bringup'),
        'launch',
        'realsense.launch.py'
    )

    velodyne_launch = os.path.join(
        get_package_share_directory('sensors_bringup'),
        'launch',
        'velodyne.launch.py'
    )
    
    gps_launch = os.path.join(
        get_package_share_directory('sensors_bringup'),
        'launch',
        'gps.launch.py'
    )

    # --- Define actions ---
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch)
    )

    velodyne = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(velodyne_launch)
    )

    gps = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gps_launch)
    )

    launch_sequence = [
        realsense,
        TimerAction(period=2.0, actions=[velodyne]),
        TimerAction(period=2.0, actions=[gps]),    
    ]
    
    return LaunchDescription(launch_sequence)
