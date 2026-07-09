import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    sensors_bringup_dir = get_package_share_directory("sensors_bringup")
    robot_nav_dir = get_package_share_directory("robot_navigation")

    target_prompt = LaunchConfiguration("target_prompt")

    declare_target_prompt = DeclareLaunchArgument(
        "target_prompt",
        default_value="person",
        description="CLIPSeg prompt to segment, e.g. person, chair, backpack",
    )
    
    clipseg_debug_launch = os.path.join(
        robot_nav_dir,
        "launch",
        "clipseg_debug_image_compressed.launch.py",
    )

    clipseg_debug = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(clipseg_debug_launch),
    )

    base_to_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera_tf",
        arguments=[
            "0.125", "-0.15", "0.825",
            "0", "0", "0",
            "panther/base_link",
            "camera_link",
        ],
        output="screen",
    )

    camera_to_depth_optical = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_to_depth_optical_tf",
        arguments=[
            "0", "0", "0",
            "-1.57079632679", "0", "-1.57079632679",
            "camera_link",
            "camera_depth_optical_frame",
        ],
        output="screen",
    )

    clipseg = Node(
        package="semantic_landmarks",
        executable="clipseg_depth_node_with_multimasks",
        name="clipseg_depth_node",
        output="screen",
        parameters=[
            {
                "rgb_topic": "/camera/color/image_compressed",
                "depth_topic": "/camera/depth/image_compressed",
                "prompts": target_prompt,
                "mask_threshold": 0.5,
            }
        ],
    )

    landmark_logger = Node(
        package="semantic_landmarks",
        executable="masked_depth_landmark_logger",
        name="masked_depth_landmark_logger",
        output="screen",
        parameters=[
            {
                "prompts": target_prompt,
                "masked_depth_prefix": "/clipseg/masked_depth",
                "camera_info_topic": "/camera/depth/camera_info_compressed",
                "target_point_prefix": "/target_point",
                "output_file": "/rgbd_camera_intel_dev/landmarks.yaml",
                "write_period_sec": 5.0,
                "min_depth_m": 0.20,
                "max_depth_m": 10.0,
                "min_valid_pixels": 50,
                "samples_window": 50,
            }
        ],
    )


    return LaunchDescription([
        declare_target_prompt,

        # Static TFs first.
        TimerAction(period=0.5, actions=[
            base_to_camera,
            camera_to_depth_optical,
        ]),

        # Perception after RealSense is up.
        TimerAction(period=7.0, actions=[
            clipseg,
        ]),
        
        TimerAction(period=8.0, actions=[
            clipseg_debug,
        ]),

        TimerAction(period=9.0, actions=[
            landmark_logger,
        ]),
    ])