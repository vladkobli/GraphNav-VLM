import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
        # ===================== LAUNCH ARGUMENTS =====================
    realsense_arg = DeclareLaunchArgument(
        'realsense',
        default_value='true'
    )

    velodyne_arg = DeclareLaunchArgument(
        'velodyne',
        default_value='true'
    )

    gps_arg = DeclareLaunchArgument(
        'gps',
        default_value='true'
    )

    nav2_arg = DeclareLaunchArgument(
        'nav2',
        default_value='true'
    )

    # ===================== LAUNCH FILE PATHS =====================
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

    nav2_launch = os.path.join(
        get_package_share_directory('sensors_bringup'),
        'launch',
        'nav2.launch.py'
    )

    # ===================== CONDITIONAL NODES =====================
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        condition=IfCondition(LaunchConfiguration('realsense'))
    )

    velodyne = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(velodyne_launch),
        condition=IfCondition(LaunchConfiguration('velodyne'))
    )

    gps = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gps_launch),
        condition=IfCondition(LaunchConfiguration('gps'))
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch),
        condition=IfCondition(LaunchConfiguration('nav2'))
    )


    return LaunchDescription([
        realsense_arg,
        velodyne_arg,
        gps_arg,
        nav2_arg,

        realsense,
        TimerAction(period=2.0, actions=[velodyne]),
        TimerAction(period=2.0, actions=[gps]),
        TimerAction(period=2.0, actions=[nav2])
    ])
