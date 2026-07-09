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

    use_lidar = LaunchConfiguration("use_lidar")
    use_realsense = LaunchConfiguration("use_realsense")
    use_slam = LaunchConfiguration("use_slam")
    use_nav2 = LaunchConfiguration("use_nav2")
    target_prompt = LaunchConfiguration("target_prompt")

    declare_use_lidar = DeclareLaunchArgument(
        "use_lidar",
        default_value="true",
        description="Start Velodyne LiDAR launch",
    )

    declare_use_realsense = DeclareLaunchArgument(
        "use_realsense",
        default_value="true",
        description="Start RealSense launch",
    )

    declare_use_slam = DeclareLaunchArgument(
        "use_slam",
        default_value="true",
        description="Start SLAM Toolbox. This publishes map -> panther/odom.",
    )

    declare_use_nav2 = DeclareLaunchArgument(
        "use_nav2",
        default_value="true",
        description="Start Nav2 target-following launch",
    )

    declare_target_prompt = DeclareLaunchArgument(
        "target_prompt",
        default_value="person",
        description="CLIPSeg prompt to segment, e.g. person, chair, backpack",
    )

    velodyne_launch = os.path.join(
        sensors_bringup_dir,
        "launch",
        "velodyne.launch.py",
    )

    realsense_launch = os.path.join(
        sensors_bringup_dir,
        "launch",
        "realsense.launch.py",
    )
    
    downsample_launch = os.path.join(
        robot_nav_dir,
        "launch",
        "downsampled_realsense.launch.py",
    )

    slam_launch = os.path.join(
        robot_nav_dir,
        "launch",
        "slam.launch.py",
    )

    nav2_launch = os.path.join(
        robot_nav_dir,
        "launch",
        "target_follow.launch.py",
    )
    
    clipseg_debug_launch = os.path.join(
        robot_nav_dir,
        "launch",
        "clipseg_debug_image_compressed.launch.py",
    )

    velodyne = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(velodyne_launch),
        condition=IfCondition(use_lidar),
    )

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        condition=IfCondition(use_realsense),
    )
    
    downsample = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(downsample_launch),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch),
        condition=IfCondition(use_slam),
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

    base_to_velodyne = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_velodyne_tf",
        arguments=[
            "0.125", "0.02", "0.825",
            "0", "0", "0",
            "panther/base_link",
            "panther/velodyne_link",
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

    rgbd2pointcloud = Node(
        package="robot_navigation",
        executable="rgbd2pointcloud",
        name="rgbd2pointcloud",
        output="screen",
    )

    crop_lidar = Node(
        package="robot_navigation",
        executable="crop_lidar",
        name="crop_lidar",
        output="screen",
        condition=IfCondition(use_lidar),
        parameters=[
            {
                "roi_timeout_sec": 5.0,
                "padding_x": 0.50,
                "padding_y": 0.50,
                "padding_z_down": 1.10,
                "padding_z_up": 0.20,
                "target_frame": "panther/base_link",
            }
        ],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch),
        condition=IfCondition(use_nav2),
    )

    return LaunchDescription([
        declare_use_lidar,
        declare_use_realsense,
        declare_use_slam,
        declare_use_nav2,
        declare_target_prompt,

        # Static TFs first.
        TimerAction(period=0.5, actions=[
            base_to_camera,
            camera_to_depth_optical,
            base_to_velodyne,
        ]),

        # LiDAR first because SLAM needs /scan.
        TimerAction(period=1.0, actions=[
            velodyne,
        ]),

        # RealSense can start independently.
        TimerAction(period=2.0, actions=[
            realsense,
        ]),
        
        # RealSense can start independently.
        TimerAction(period=4.0, actions=[
            downsample,
        ]),

        # SLAM starts after LiDAR/scan has a moment to appear.
        # SLAM publishes map -> panther/odom.
        TimerAction(period=4.0, actions=[
            slam,
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

        TimerAction(period=10.0, actions=[
            rgbd2pointcloud,
        ]),

        TimerAction(period=11.0, actions=[
            crop_lidar,
        ]),

        # Nav2 last, after SLAM has had time to publish map -> odom.
        TimerAction(period=14.0, actions=[
            nav2,
        ]),
    ])