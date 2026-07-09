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

    use_lidar = LaunchConfiguration("use_lidar")
    use_slam = LaunchConfiguration("use_slam")
    use_rviz = LaunchConfiguration("use_rviz")
    use_logger = LaunchConfiguration("use_logger")
    nodes_dir = LaunchConfiguration("nodes_dir")
    rviz_config = LaunchConfiguration("rviz_config")

    lerobot_server_url = LaunchConfiguration("lerobot_server_url")
    camera_tf_source = LaunchConfiguration("camera_tf_source")
    lerobot_state_path = LaunchConfiguration("lerobot_state_path")
    lerobot_yaw_path = LaunchConfiguration("lerobot_yaw_path")
    lerobot_position_path = LaunchConfiguration("lerobot_position_path")
    lerobot_position_zero = LaunchConfiguration("lerobot_position_zero")
    lerobot_position_to_rad = LaunchConfiguration("lerobot_position_to_rad")
    lerobot_joint_state_topic = LaunchConfiguration("lerobot_joint_state_topic")
    lerobot_yaw_joint_name = LaunchConfiguration("lerobot_yaw_joint_name")
    realsense_color_topic = LaunchConfiguration("realsense_color_topic")
    realsense_depth_topic = LaunchConfiguration("realsense_depth_topic")

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

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_lerobot_realsense_sequence_logging",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
    )

    logger = Node(
        package="camera_capture",
        executable="lerobot_realsense_sequence_logger",
        name="lerobot_realsense_sequence_logger",
        output="screen",
        parameters=[{
            "nodes_dir": nodes_dir,
            "global_frame": "map",
            "robot_frame": "panther/base_link",

            "lerobot_server_url": lerobot_server_url,

            # 8 semantic camera stops. The logger sends these as stop_index/stop_name
            # to the LeRobot HTTP server. The server maps them to raw m5 positions.
            "camera_stop_names": [
                "back",
                "back_left",
                "left",
                "front_left",
                "front",
                "front_right",
                "right",
                "back_right",
            ],

            "settle_after_motion_sec": 0.5,
            "wait_for_fresh_frame_sec": 60.0,
            "fresh_frames_to_skip": 0,

            "realsense_color_topic": realsense_color_topic,
            "realsense_aligned_depth_topic": realsense_depth_topic,
            "require_all_topics": True,
        }],
        condition=IfCondition(use_logger),
    )

    lerobot_camera_tf = Node(
        package="camera_capture",
        executable="lerobot_camera_tf_broadcaster",
        name="lerobot_camera_tf_broadcaster",
        output="screen",
        parameters=[{
            "source": camera_tf_source,
            "parent_frame": "panther/body_link",
            "child_frame": "camera_link",
            "xyz": [-0.125, 0.02, 0.818],
            "roll": 0.0,
            "pitch": 0.0,
            "yaw_offset": 0.0,
            "lerobot_server_url": lerobot_server_url,
            "http_state_path": lerobot_state_path,
            "http_yaw_path": lerobot_yaw_path,
            "http_position_path": lerobot_position_path,
            "position_zero": lerobot_position_zero,
            "position_to_rad": lerobot_position_to_rad,
            "joint_state_topic": lerobot_joint_state_topic,
            "joint_name": lerobot_yaw_joint_name,
            "publish_rate_hz": 20.0,
        }],
    )

    tf_nodes = [
        static_tf(
            "base_to_velodyne_tf",
            (0.125, 0.02, 0.643),
            0.0,
            "panther/body_link",
            "panther/velodyne_link",
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("use_lidar", default_value="false"),
        DeclareLaunchArgument("use_slam", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("use_logger", default_value="true"),

        DeclareLaunchArgument("nodes_dir", default_value="/rgbd_camera_intel_dev/src/nodes"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value="/rgbd_camera_intel_dev/src/camera_capture/rviz/dataset_logging2.rviz",
        ),
        DeclareLaunchArgument(
            "lerobot_server_url",
            default_value="http://host.docker.internal:8765",
        ),
        DeclareLaunchArgument("camera_tf_source", default_value="http"),
        DeclareLaunchArgument("lerobot_state_path", default_value="/health"),
        DeclareLaunchArgument("lerobot_yaw_path", default_value=""),
        DeclareLaunchArgument("lerobot_position_path", default_value="positions.m5"),
        DeclareLaunchArgument("lerobot_position_zero", default_value="2233.0"),
        DeclareLaunchArgument("lerobot_position_to_rad", default_value="-0.001525045"),
        DeclareLaunchArgument("lerobot_joint_state_topic", default_value="/lerobot/joint_states"),
        DeclareLaunchArgument("lerobot_yaw_joint_name", default_value="m5"),
        DeclareLaunchArgument(
            "realsense_color_topic",
            default_value="/camera/driver/color/image_raw",
        ),
        DeclareLaunchArgument(
            "realsense_depth_topic",
            default_value="/camera/driver/aligned_depth_to_color/image_raw",
        ),

        TimerAction(period=0.5, actions=tf_nodes),
        TimerAction(period=0.5, actions=[lerobot_camera_tf]),
        TimerAction(period=1.0, actions=[velodyne]),
        TimerAction(period=3.0, actions=[slam]),
        TimerAction(period=6.0, actions=[rviz, logger]),
    ])
