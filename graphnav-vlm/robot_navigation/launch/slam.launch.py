import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    slam_toolbox_dir = get_package_share_directory("slam_toolbox")
    robot_nav_dir = get_package_share_directory("robot_navigation")

    default_slam_params_file = os.path.join(
        robot_nav_dir,
        "config",
        "slam_toolbox_params.yaml",
    )
    slam_params_file = LaunchConfiguration("slam_params_file")

    return LaunchDescription([
        DeclareLaunchArgument("slam_params_file", default_value=default_slam_params_file),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    slam_toolbox_dir,
                    "launch",
                    "online_async_launch.py",
                )
            ),
            launch_arguments={
                "use_sim_time": "false",
                "slam_params_file": slam_params_file,
            }.items(),
        )
    ])
