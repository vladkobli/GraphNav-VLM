import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def static_tf(name, xyz, yaw, parent, child, pitch=0.0, roll=0.0):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            f"{xyz[0]}",
            f"{xyz[1]}",
            f"{xyz[2]}",
            f"{yaw}",
            f"{pitch}",
            f"{roll}",
            parent,
            child,
        ],
        output="screen",
    )


def generate_launch_description():
    sensors_bringup_dir = get_package_share_directory("sensors_bringup")
    robot_nav_dir = get_package_share_directory("robot_navigation")
    camera_capture_dir = get_package_share_directory("camera_capture")

    use_lidar = LaunchConfiguration("use_lidar")
    use_slam = LaunchConfiguration("use_slam")
    use_nav2 = LaunchConfiguration("use_nav2")
    use_rviz = LaunchConfiguration("use_rviz")
    use_waypoint_logger = LaunchConfiguration("use_waypoint_logger")
    nodes_dir = LaunchConfiguration("nodes_dir")
    rviz_config = LaunchConfiguration("rviz_config")

    velodyne = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensors_bringup_dir, "launch", "velodyne.launch.py")
        ),
        condition=IfCondition(use_lidar),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_dir, "launch", "slam.launch.py")
        ),
        condition=IfCondition(use_slam),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(camera_capture_dir, "launch", "nav2_waypoint_logging.launch.py")
        ),
        condition=IfCondition(use_nav2),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_waypoint_logging",
        output="screen",
        remappings=[
            ("/goal_pose", "/waypoint_logging/goal_pose"),
            ("goal_pose", "/waypoint_logging/goal_pose"),
        ],
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
    )

    waypoint_logger = Node(
        package="camera_capture",
        executable="waypoint_dataset_logger",
        name="waypoint_dataset_logger",
        output="screen",
        parameters=[{
            "nodes_dir": nodes_dir,
            "save_dir": nodes_dir,
            "waypoint_topic": "/waypoint_logging/goal_pose",
            "start_topic": "/waypoint_logging/start",
            "reset_topic": "/waypoint_logging/reset",
            "remove_nearest_topic": "/waypoint_logging/remove_nearest",
            "global_frame": "map",
            "robot_frame": "panther/base_link",
            "arrival_wait_sec": 2.4,
            "post_snapshot_wait_sec": 0.0,
            "snapshot_yaw_offsets_degrees": [0.0, 30.0, 60.0],
            "odom_topic": "/odometry/local",
            "stopped_linear_velocity": 0.05,
            "stopped_angular_velocity": 0.08,
            "stop_wait_timeout_sec": 1.5,
            "stop_cmd_vel_topic": "/panther/cmd_vel",
            "stop_command_duration_sec": 0.25,
            "require_all_topics": False,
        }],
        condition=IfCondition(use_waypoint_logger),
    )

    tf_nodes = [
        static_tf(
            "base_to_velodyne_tf",
            (0.125, 0.02, 0.643),
            0.0,
            "panther/body_link",
            "panther/velodyne_link",
        ),
        static_tf(
            "base_to_camera2_tf",
            (0.175, -0.09, 0.52),
            math.radians(0.0),
            "panther/body_link",
            "camera2_link",
        ),
        static_tf(
            "base_to_camera3_tf",
            (0.0, -0.265, 0.52),
            math.radians(270.0),
            "panther/body_link",
            "camera3_link",
        ),
        static_tf(
            "base_to_camera1_tf",
            (-0.175, -0.09, 0.52),
            math.radians(180.0),
            "panther/body_link",
            "camera1_link",
        ),
        static_tf(
            "base_to_camera0_tf",
            (0.0, -0.08, 0.52),
            math.radians(90.0),
            "panther/body_link",
            "camera0_link",
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("use_lidar", default_value="true"),
        DeclareLaunchArgument("use_slam", default_value="true"),
        DeclareLaunchArgument("use_nav2", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_waypoint_logger", default_value="true"),
        DeclareLaunchArgument("nodes_dir", default_value="/rgbd_camera_intel_dev/src/nodes"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value="/rgbd_camera_intel_dev/src/dataset_logging2.rviz",
        ),

        TimerAction(period=0.5, actions=tf_nodes),
        TimerAction(period=1.0, actions=[velodyne]),
        TimerAction(period=3.0, actions=[slam]),
        TimerAction(period=6.0, actions=[nav2]),
        TimerAction(period=8.0, actions=[rviz, waypoint_logger]),
    ])
