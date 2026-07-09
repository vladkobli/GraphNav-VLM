import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def static_tf(name, xyz, rpy, parent, child):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            str(xyz[0]), str(xyz[1]), str(xyz[2]),
            str(rpy[0]), str(rpy[1]), str(rpy[2]),
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
    use_nav2 = LaunchConfiguration("use_nav2")
    use_3d_lidar_frontier = LaunchConfiguration("use_3d_lidar_frontier")
    use_startup_360_scan = LaunchConfiguration("use_startup_360_scan")
    use_frontier_detector = LaunchConfiguration("use_frontier_detector")
    use_frontier_marker_probe = LaunchConfiguration("use_frontier_marker_probe")
    use_frontier_client = LaunchConfiguration("use_frontier_client")
    use_rviz = LaunchConfiguration("use_rviz")
    auto_start = LaunchConfiguration("auto_start")
    rviz_config = LaunchConfiguration("rviz_config")
    startup_scan_cmd_vel_topic = LaunchConfiguration("startup_scan_cmd_vel_topic")
    startup_scan_direct_cmd_vel_topic = LaunchConfiguration("startup_scan_direct_cmd_vel_topic")
    startup_scan_angular_speed = LaunchConfiguration("startup_scan_angular_speed")
    startup_scan_rotations = LaunchConfiguration("startup_scan_rotations")
    startup_scan_start_delay_sec = LaunchConfiguration("startup_scan_start_delay_sec")
    startup_scan_settle_sec = LaunchConfiguration("startup_scan_settle_sec")
    frontier_startup_delay_sec = LaunchConfiguration("frontier_startup_delay_sec")
    frontier_region_size_thresh = LaunchConfiguration("frontier_region_size_thresh")
    frontier_preprocess_iterations = LaunchConfiguration("frontier_preprocess_iterations")
    frontier_marker_scale = LaunchConfiguration("frontier_marker_scale")
    frontier_marker_min_region_size = LaunchConfiguration("frontier_marker_min_region_size")
    frontier_marker_max_count = LaunchConfiguration("frontier_marker_max_count")
    frontier_nav2_params_file = LaunchConfiguration("frontier_nav2_params_file")
    frontier_slam_params_file = LaunchConfiguration("frontier_slam_params_file")
    lidar_3d_input_topic = LaunchConfiguration("lidar_3d_input_topic")
    lidar_3d_filtered_topic = LaunchConfiguration("lidar_3d_filtered_topic")
    lidar_3d_scan_topic = LaunchConfiguration("lidar_3d_scan_topic")
    lidar_3d_z_min = LaunchConfiguration("lidar_3d_z_min")
    lidar_3d_z_max = LaunchConfiguration("lidar_3d_z_max")
    lidar_3d_range_max = LaunchConfiguration("lidar_3d_range_max")

    # Optional: only enable if Panther bringup is NOT already starting the lidar.
    velodyne = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensors_bringup_dir, "launch", "velodyne.launch.py")
        ),
        condition=IfCondition(use_lidar),
    )

    # This must publish map -> panther/odom.
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_dir, "launch", "slam.launch.py")
        ),
        launch_arguments={
            "slam_params_file": frontier_slam_params_file,
        }.items(),
        condition=IfCondition(use_slam),
    )

    pointcloud_height_filter = Node(
        package="robot_navigation",
        executable="pointcloud_height_filter",
        name="frontier_pointcloud_height_filter",
        output="screen",
        parameters=[{
            "input_topic": lidar_3d_input_topic,
            "filtered_cloud_topic": lidar_3d_filtered_topic,
            "scan_topic": lidar_3d_scan_topic,
            "target_frame": "panther/base_link",
            "z_min": lidar_3d_z_min,
            "z_max": lidar_3d_z_max,
            "range_min": 0.25,
            "range_max": lidar_3d_range_max,
        }],
        condition=IfCondition(use_3d_lidar_frontier),
    )

    # General Nav2 launch. It must expose /navigate_to_pose.
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_dir, "launch", "nav2.launch.py")
        ),
        launch_arguments={
            "params_file": frontier_nav2_params_file,
        }.items(),
        condition=IfCondition(use_nav2),
    )

    classical_frontier_detector = Node(
        package="frontier_exploration",
        executable="classical_frontier_detector",
        name="classical_frontier_detector",
        output="screen",
        parameters=[{
            "region_size_thresh": frontier_region_size_thresh,
            "preprocess_iterations": frontier_preprocess_iterations,
            "robot_width": 0.85,
            "marker_scale": frontier_marker_scale,
            "marker_min_region_size": frontier_marker_min_region_size,
            "marker_max_count": frontier_marker_max_count,
            "occupancy_map_msg": "/map",
        }],
        condition=IfCondition(use_frontier_detector),
    )

    startup_360_scan = Node(
        package="robot_navigation",
        executable="startup_360_scan",
        name="startup_360_scan",
        output="screen",
        parameters=[{
            "enabled": use_startup_360_scan,
            "cmd_vel_topic": startup_scan_cmd_vel_topic,
            "direct_cmd_vel_topic": startup_scan_direct_cmd_vel_topic,
            "robot_frame": "panther/base_link",
            "angular_speed": startup_scan_angular_speed,
            "rotations": startup_scan_rotations,
            "start_delay_sec": startup_scan_start_delay_sec,
            "settle_sec": startup_scan_settle_sec,
        }],
        condition=IfCondition(use_startup_360_scan),
    )

    frontier_marker_probe = Node(
        package="robot_navigation",
        executable="panther_frontier_marker_probe",
        name="panther_frontier_marker_probe",
        output="screen",
        parameters=[{
            "frontier_service": "/frontier_pose",
            "refresh_period_sec": 2.0,
            "goal_rank": 0,
        }],
        condition=IfCondition(use_frontier_marker_probe),
    )

    panther_frontier_client = Node(
        package="robot_navigation",
        executable="panther_frontier_nav2_client",
        name="panther_frontier_nav2_client",
        output="screen",
        parameters=[{
            "global_frame": "map",
            "robot_frame": "panther/base_link",
            "map_topic": "/map",
            "frontier_service": "/frontier_pose",
            "nav2_action": "/navigate_to_pose",
            "auto_start": auto_start,
            "nav_result_timeout_sec": 75.0,
            "timer_period_sec": 2.0,
            "startup_delay_sec": frontier_startup_delay_sec,
            "min_goal_distance": 0.7,
        }],
        condition=IfCondition(use_frontier_client),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_panther_frontier_exploration",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
    )

    tf_nodes = [
        # Use the same velodyne transform as your working pipeline.
        static_tf(
            "base_to_velodyne_tf",
            (0.125, 0.02, 0.643),
            (0.0, 0.0, 0.0),
            "panther/base_link",
            "panther/velodyne_link",
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("use_lidar", default_value="true"),
        DeclareLaunchArgument("use_slam", default_value="true"),
        DeclareLaunchArgument("use_nav2", default_value="true"),
        DeclareLaunchArgument("use_3d_lidar_frontier", default_value="true"),
        DeclareLaunchArgument("use_startup_360_scan", default_value="true"),
        DeclareLaunchArgument("use_frontier_detector", default_value="true"),
        DeclareLaunchArgument("use_frontier_marker_probe", default_value="true"),
        DeclareLaunchArgument("use_frontier_client", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("auto_start", default_value="false"),
        DeclareLaunchArgument("startup_scan_cmd_vel_topic", default_value="/cmd_vel_nav"),
        DeclareLaunchArgument("startup_scan_direct_cmd_vel_topic", default_value="/panther/cmd_vel"),
        DeclareLaunchArgument("startup_scan_angular_speed", default_value="0.35"),
        DeclareLaunchArgument("startup_scan_rotations", default_value="1.0"),
        DeclareLaunchArgument("startup_scan_start_delay_sec", default_value="1.0"),
        DeclareLaunchArgument("startup_scan_settle_sec", default_value="2.0"),
        DeclareLaunchArgument("frontier_startup_delay_sec", default_value="22.0"),
        DeclareLaunchArgument("frontier_region_size_thresh", default_value="12"),
        DeclareLaunchArgument("frontier_preprocess_iterations", default_value="0"),
        DeclareLaunchArgument("frontier_marker_scale", default_value="0.08"),
        DeclareLaunchArgument("frontier_marker_min_region_size", default_value="12"),
        DeclareLaunchArgument("frontier_marker_max_count", default_value="35"),
        DeclareLaunchArgument(
            "frontier_nav2_params_file",
            default_value=os.path.join(
                robot_nav_dir,
                "config",
                "nav2_params_frontier_exploration.yaml",
            ),
        ),
        DeclareLaunchArgument(
            "frontier_slam_params_file",
            default_value=os.path.join(
                robot_nav_dir,
                "config",
                "slam_toolbox_params_frontier_3d.yaml",
            ),
        ),
        DeclareLaunchArgument("lidar_3d_input_topic", default_value="/velodyne_points"),
        DeclareLaunchArgument("lidar_3d_filtered_topic", default_value="/velodyne_points_without_roi"),
        DeclareLaunchArgument("lidar_3d_scan_topic", default_value="/scan_frontier_3d"),
        DeclareLaunchArgument("lidar_3d_z_min", default_value="0.3"),
        DeclareLaunchArgument("lidar_3d_z_max", default_value="2.0"),
        DeclareLaunchArgument("lidar_3d_range_max", default_value="20.0"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=os.path.join(
                robot_nav_dir,
                "rviz",
                "panther_frontier_exploration.rviz",
            ),
        ),

        # Static TFs first. SLAM needs panther/base_link -> panther/velodyne_link.
        TimerAction(period=0.5, actions=tf_nodes),

        # Lidar first if this launch is responsible for it.
        TimerAction(period=1.0, actions=[velodyne]),

        # Filter/project 3D LiDAR before SLAM consumes the generated scan.
        TimerAction(period=2.0, actions=[pointcloud_height_filter]),

        # SLAM must start before Nav2/frontier. It publishes map -> panther/odom.
        TimerAction(period=3.0, actions=[slam]),

        # Nav2 after map->odom has time to appear.
        TimerAction(period=6.0, actions=[nav2]),

        # Do an in-place scan before autonomous frontier movement.
        TimerAction(period=8.0, actions=[startup_360_scan]),

        # Frontier after /map and Nav2 exist.
        TimerAction(period=10.0, actions=[classical_frontier_detector]),
        TimerAction(period=11.0, actions=[frontier_marker_probe]),
        TimerAction(period=12.0, actions=[panther_frontier_client]),

        # RViz last.
        TimerAction(period=13.0, actions=[rviz]),
    ])
