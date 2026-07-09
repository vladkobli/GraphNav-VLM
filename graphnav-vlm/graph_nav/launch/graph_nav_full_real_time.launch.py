import os
import json
from datetime import datetime, timezone
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


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


def store_task_prompt(context, *_, **__):
    nodes_dir = Path(LaunchConfiguration("nodes_dir").perform(context))
    task_file_name = LaunchConfiguration("task_file_name").perform(context)
    task_path = nodes_dir / task_file_name
    task_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_prompt": LaunchConfiguration("user_prompt").perform(context),
        "target_prompt": LaunchConfiguration("target_prompt").perform(context),
    }
    task_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[graph_nav_full.launch.py] Stored task prompt in {task_path}")
    return []


def generate_launch_description():
    graph_nav_dir = Path(__file__).resolve().parents[1]
    sensors_bringup_dir = get_package_share_directory("sensors_bringup")
    robot_nav_dir = get_package_share_directory("robot_navigation")

    use_lidar = LaunchConfiguration("use_lidar")
    use_realsense = LaunchConfiguration("use_realsense")
    use_slam = LaunchConfiguration("use_slam")
    use_nav2 = LaunchConfiguration("use_nav2")
    use_clipseg = LaunchConfiguration("use_clipseg")
    use_rviz = LaunchConfiguration("use_rviz")
    use_graph_nav_state_machine = LaunchConfiguration("use_graph_nav_state_machine")
    use_lerobot_server = LaunchConfiguration("use_lerobot_server")
    use_lerobot_camera_tf = LaunchConfiguration("use_lerobot_camera_tf")
    use_task_memory = LaunchConfiguration("use_task_memory")
    target_prompt = LaunchConfiguration("target_prompt")
    user_prompt = LaunchConfiguration("user_prompt")
    mask_threshold = LaunchConfiguration("mask_threshold")
    use_logger = LaunchConfiguration("use_logger")
    nodes_dir = LaunchConfiguration("nodes_dir")
    
    lerobot_server_url = LaunchConfiguration("lerobot_server_url")
    lerobot_server_python = LaunchConfiguration("lerobot_server_python")
    lerobot_server_script = LaunchConfiguration("lerobot_server_script")
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
    realsense_depth_camera_info_topic = LaunchConfiguration("realsense_depth_camera_info_topic")
    max_depth_preview_m = LaunchConfiguration("max_depth_preview_m")
    moondream_server_url = LaunchConfiguration("moondream_server_url")
    require_moondream_descriptions = LaunchConfiguration("require_moondream_descriptions")
    moondream_overwrite_descriptions = LaunchConfiguration("moondream_overwrite_descriptions")
    rviz_config = LaunchConfiguration("rviz_config")
    graph_nav_state_file_name = LaunchConfiguration("graph_nav_state_file_name")
    graph_nav_phase_topic = LaunchConfiguration("graph_nav_phase_topic")
    graph_nav_finished_topic = LaunchConfiguration("graph_nav_finished_topic")
    graph_nav_target_visible_topic = LaunchConfiguration("graph_nav_target_visible_topic")
    graph_nav_goal_reached_topic = LaunchConfiguration("graph_nav_goal_reached_topic")
    graph_nav_node_reached_topic = LaunchConfiguration("graph_nav_node_reached_topic")
    target_point_topic = LaunchConfiguration("target_point_topic")
    target_visible_from_target_point = LaunchConfiguration("target_visible_from_target_point")
    target_visible_min_observations = LaunchConfiguration("target_visible_min_observations")
    target_visible_window_sec = LaunchConfiguration("target_visible_window_sec")
    target_visible_min_range_m = LaunchConfiguration("target_visible_min_range_m")
    target_visible_max_range_m = LaunchConfiguration("target_visible_max_range_m")
    target_visible_max_spread_m = LaunchConfiguration("target_visible_max_spread_m")
    stop_scan_on_target_visible = LaunchConfiguration("stop_scan_on_target_visible")
    masked_depth_min_valid_pixels = LaunchConfiguration("masked_depth_min_valid_pixels")
    masked_depth_min_depth_m = LaunchConfiguration("masked_depth_min_depth_m")
    masked_depth_max_depth_m = LaunchConfiguration("masked_depth_max_depth_m")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    graph_nav_tick_sec = LaunchConfiguration("graph_nav_tick_sec")
    settle_after_motion_sec = LaunchConfiguration("settle_after_motion_sec")
    settle_before_sweep_sec = LaunchConfiguration("settle_before_sweep_sec")
    fresh_frames_to_skip = LaunchConfiguration("fresh_frames_to_skip")
    wait_for_fresh_color_frame_sec = LaunchConfiguration("wait_for_fresh_color_frame_sec")
    candidate_top_k = LaunchConfiguration("candidate_top_k")
    candidate_memory_nodes = LaunchConfiguration("candidate_memory_nodes")
    next_node_goal_topic = LaunchConfiguration("next_node_goal_topic")
    nav2_goal_pose_topic = LaunchConfiguration("nav2_goal_pose_topic")
    publish_nav2_goal_pose = LaunchConfiguration("publish_nav2_goal_pose")
    next_node_goal_distance_m = LaunchConfiguration("next_node_goal_distance_m")
    nav2_status_topic = LaunchConfiguration("nav2_status_topic")
    auto_scan_after_nav2_success = LaunchConfiguration("auto_scan_after_nav2_success")
    auto_finish_on_goal_nav2_success = LaunchConfiguration("auto_finish_on_goal_nav2_success")
    nav2_success_min_wait_sec = LaunchConfiguration("nav2_success_min_wait_sec")
    llm_candidate_selection_enabled = LaunchConfiguration("llm_candidate_selection_enabled")
    llm_model = LaunchConfiguration("llm_model")
    ollama_base_url = LaunchConfiguration("ollama_base_url")
    llm_request_timeout_sec = LaunchConfiguration("llm_request_timeout_sec")
    llm_candidate_num_predict = LaunchConfiguration("llm_candidate_num_predict")
    
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

    declare_use_clipseg = DeclareLaunchArgument(
        "use_clipseg",
        default_value="true",
        description="Start CLIPSeg perception and target point nodes",
    )

    declare_use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Start RViz2 with the graph navigation visualization",
    )

    declare_use_graph_nav_state_machine = DeclareLaunchArgument(
        "use_graph_nav_state_machine",
        default_value="true",
        description="Start the graph_nav three-phase pipeline state-machine skeleton",
    )

    declare_use_lerobot_server = DeclareLaunchArgument(
        "use_lerobot_server",
        default_value="true",
        description="Start graph_nav LeRobot m5 sweep HTTP server",
    )

    declare_use_lerobot_camera_tf = DeclareLaunchArgument(
        "use_lerobot_camera_tf",
        default_value="true",
        description="Start LeRobot camera TF broadcaster",
    )

    declare_use_task_memory = DeclareLaunchArgument(
        "use_task_memory",
        default_value="true",
        description="Store the natural-language task prompt for graph navigation",
    )

    declare_user_prompt = DeclareLaunchArgument(
        "user_prompt",
        default_value="Find the requested object.",
        description="Natural-language user task sentence stored by graph_nav",
    )

    declare_target_prompt = DeclareLaunchArgument(
        "target_prompt",
        default_value="person",
        description="CLIPSeg prompt to segment, e.g. person, chair, backpack",
    )

    declare_mask_threshold = DeclareLaunchArgument(
        "mask_threshold",
        default_value="0.6",
        description="CLIPSeg mask threshold (0.0 - 1.0)",
    )
    
    declare_use_logger = DeclareLaunchArgument(
        "use_logger",
        default_value="true",
        description="Start Realsense sequence logger",
    )

    declare_nodes_dir = DeclareLaunchArgument(
        "nodes_dir",
        default_value="/rgbd_camera_intel_dev/src/nodes",
        description="Directory where graph nodes, camera sweeps, and task memory are stored",
    )

    declare_task_file_name = DeclareLaunchArgument(
        "task_file_name",
        default_value="graph_nav_task.json",
        description="Task metadata filename written inside nodes_dir",
    )

    declare_lerobot_server_url = DeclareLaunchArgument(
        "lerobot_server_url",
        default_value="http://127.0.0.1:8765",
        description="HTTP URL for the LeRobot camera sweep server",
    )

    declare_lerobot_server_python = DeclareLaunchArgument(
        "lerobot_server_python",
        default_value="python3",
        description="Python executable used to start the LeRobot sweep server",
    )

    declare_lerobot_server_script = DeclareLaunchArgument(
        "lerobot_server_script",
        default_value=str(graph_nav_dir / "graph_nav" / "lerobot_camera_sweep_server.py"),
        description="Path to the LeRobot sweep server script",
    )

    declare_camera_tf_source = DeclareLaunchArgument(
        "camera_tf_source",
        default_value="http",
        description="Camera TF source used by lerobot_camera_tf_broadcaster",
    )

    declare_lerobot_state_path = DeclareLaunchArgument(
        "lerobot_state_path",
        default_value="/health",
    )

    declare_lerobot_yaw_path = DeclareLaunchArgument(
        "lerobot_yaw_path",
        default_value="",
    )

    declare_lerobot_position_path = DeclareLaunchArgument(
        "lerobot_position_path",
        default_value="positions.m5",
    )

    declare_lerobot_position_zero = DeclareLaunchArgument(
        "lerobot_position_zero",
        default_value="2233.0",
    )

    declare_lerobot_position_to_rad = DeclareLaunchArgument(
        "lerobot_position_to_rad",
        default_value="-0.001525045",
    )

    declare_lerobot_joint_state_topic = DeclareLaunchArgument(
        "lerobot_joint_state_topic",
        default_value="/lerobot/joint_states",
    )

    declare_lerobot_yaw_joint_name = DeclareLaunchArgument(
        "lerobot_yaw_joint_name",
        default_value="m5",
    )

    declare_realsense_color_topic = DeclareLaunchArgument(
        "realsense_color_topic",
        default_value="/camera/driver/color/image_raw",
    )

    declare_realsense_depth_topic = DeclareLaunchArgument(
        "realsense_depth_topic",
        default_value="/camera/driver/aligned_depth_to_color/image_raw",
    )

    declare_realsense_depth_camera_info_topic = DeclareLaunchArgument(
        "realsense_depth_camera_info_topic",
        default_value="/camera/driver/aligned_depth_to_color/camera_info",
        description="CameraInfo topic for aligned RealSense depth",
    )

    declare_max_depth_preview_m = DeclareLaunchArgument(
        "max_depth_preview_m",
        default_value="5.0",
        description="Maximum depth in meters used for human-viewable preview PNGs",
    )

    declare_moondream_server_url = DeclareLaunchArgument(
        "moondream_server_url",
        default_value="http://127.0.0.1:8766",
        description="HTTP URL for the external Moondream description server",
    )

    declare_require_moondream_descriptions = DeclareLaunchArgument(
        "require_moondream_descriptions",
        default_value="true",
        description="Fail create_node/scan_360 if Moondream descriptions cannot be generated",
    )

    declare_moondream_overwrite_descriptions = DeclareLaunchArgument(
        "moondream_overwrite_descriptions",
        default_value="false",
        description="Regenerate Moondream descriptions even when metadata already has them",
    )

    declare_rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(robot_nav_dir, "rviz", "navigation-new2.rviz"),
        description="RViz2 config file",
    )

    declare_graph_nav_state_file_name = DeclareLaunchArgument(
        "graph_nav_state_file_name",
        default_value="graph_nav_state.json",
        description="State-machine status filename written inside nodes_dir",
    )

    declare_graph_nav_phase_topic = DeclareLaunchArgument(
        "graph_nav_phase_topic",
        default_value="/graph_nav/phase",
        description="Topic publishing the current graph navigation phase",
    )

    declare_graph_nav_finished_topic = DeclareLaunchArgument(
        "graph_nav_finished_topic",
        default_value="/graph_nav/finished",
        description="Topic publishing the FINISHED flag",
    )

    declare_graph_nav_target_visible_topic = DeclareLaunchArgument(
        "graph_nav_target_visible_topic",
        default_value="/graph_nav/target_visible",
        description="Bool topic used to exit EXPLORATION when the target is visible",
    )

    declare_graph_nav_goal_reached_topic = DeclareLaunchArgument(
        "graph_nav_goal_reached_topic",
        default_value="/graph_nav/goal_reached",
        description="Bool topic used to exit GOAL_NAVIGATION when the target is reached",
    )

    declare_graph_nav_node_reached_topic = DeclareLaunchArgument(
        "graph_nav_node_reached_topic",
        default_value="/graph_nav/node_reached",
        description="Bool topic used to trigger create_node and scan_360 after navigation",
    )

    declare_target_point_topic = DeclareLaunchArgument(
        "target_point_topic",
        default_value="/target_point",
        description="PointStamped topic published when CLIPSeg plus depth finds the target",
    )

    declare_target_visible_from_target_point = DeclareLaunchArgument(
        "target_visible_from_target_point",
        default_value="true",
        description="Use target_point_topic observations to enter GOAL_NAVIGATION",
    )

    declare_target_visible_min_observations = DeclareLaunchArgument(
        "target_visible_min_observations",
        default_value="5",
        description="Number of target_point observations required before target is considered visible",
    )

    declare_target_visible_window_sec = DeclareLaunchArgument(
        "target_visible_window_sec",
        default_value="1.5",
        description="Time window for target_visible_min_observations",
    )

    declare_target_visible_min_range_m = DeclareLaunchArgument(
        "target_visible_min_range_m",
        default_value="0.2",
        description="Minimum target_point range accepted as a visible target",
    )

    declare_target_visible_max_range_m = DeclareLaunchArgument(
        "target_visible_max_range_m",
        default_value="8.0",
        description="Maximum target_point range accepted as a visible target",
    )

    declare_target_visible_max_spread_m = DeclareLaunchArgument(
        "target_visible_max_spread_m",
        default_value="0.0",
        description="Optional max spread among target_point observations; 0 disables this check",
    )

    declare_stop_scan_on_target_visible = DeclareLaunchArgument(
        "stop_scan_on_target_visible",
        default_value="true",
        description="Interrupt scan_360 when the target becomes visible",
    )

    declare_masked_depth_min_valid_pixels = DeclareLaunchArgument(
        "masked_depth_min_valid_pixels",
        default_value="50",
        description="Minimum non-zero masked depth pixels before publishing /target_point",
    )

    declare_masked_depth_min_depth_m = DeclareLaunchArgument(
        "masked_depth_min_depth_m",
        default_value="0.20",
        description="Minimum depth used by masked_depth_target_point",
    )

    declare_masked_depth_max_depth_m = DeclareLaunchArgument(
        "masked_depth_max_depth_m",
        default_value="10.0",
        description="Maximum depth used by masked_depth_target_point",
    )

    declare_cmd_vel_topic = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Velocity command topic used to stop the robot in FINISHED",
    )

    declare_graph_nav_tick_sec = DeclareLaunchArgument(
        "graph_nav_tick_sec",
        default_value="1.0",
        description="State-machine skeleton tick period in seconds",
    )

    declare_settle_after_motion_sec = DeclareLaunchArgument(
        "settle_after_motion_sec",
        default_value="0.5",
        description="Seconds to wait after each LeRobot pan before waiting for fresh frames",
    )

    declare_settle_before_sweep_sec = DeclareLaunchArgument(
        "settle_before_sweep_sec",
        default_value="2.0",
        description="Seconds to wait at the back stop before taking the first sweep image",
    )

    declare_fresh_frames_to_skip = DeclareLaunchArgument(
        "fresh_frames_to_skip",
        default_value="1",
        description="Number of fresh post-pan color frames to skip before saving each sweep image",
    )

    declare_wait_for_fresh_color_frame_sec = DeclareLaunchArgument(
        "wait_for_fresh_color_frame_sec",
        default_value="3.0",
        description="Seconds to wait for fresh post-pan color frames before saving latest cached frame",
    )

    declare_candidate_top_k = DeclareLaunchArgument(
        "candidate_top_k",
        default_value="3",
        description="Number of ranked direction candidates stored after each scan",
    )

    declare_candidate_memory_nodes = DeclareLaunchArgument(
        "candidate_memory_nodes",
        default_value="8",
        description="Number of previous node candidate summaries used as memory",
    )

    declare_next_node_goal_topic = DeclareLaunchArgument(
        "next_node_goal_topic",
        default_value="/graph_nav/next_node_goal_pose",
        description="PoseStamped topic for the generated next graph/photo node goal",
    )

    declare_nav2_goal_pose_topic = DeclareLaunchArgument(
        "nav2_goal_pose_topic",
        default_value="/goal_pose",
        description="Optional Nav2 goal_pose topic used when publish_nav2_goal_pose is true",
    )

    declare_publish_nav2_goal_pose = DeclareLaunchArgument(
        "publish_nav2_goal_pose",
        default_value="false",
        description="Also publish the generated next-node goal directly to Nav2 /goal_pose",
    )

    declare_next_node_goal_distance_m = DeclareLaunchArgument(
        "next_node_goal_distance_m",
        default_value="2.0",
        description="Forward distance from the current graph node to the generated next photo node",
    )

    declare_nav2_status_topic = DeclareLaunchArgument(
        "nav2_status_topic",
        default_value="/navigate_to_pose/_action/status",
        description="Nav2 NavigateToPose action status topic used to trigger the next graph scan",
    )

    declare_auto_scan_after_nav2_success = DeclareLaunchArgument(
        "auto_scan_after_nav2_success",
        default_value="true",
        description="Schedule the next graph node scan when Nav2 reports the generated goal succeeded",
    )

    declare_auto_finish_on_goal_nav2_success = DeclareLaunchArgument(
        "auto_finish_on_goal_nav2_success",
        default_value="true",
        description="Set FINISHED when Nav2 reports the target-following goal succeeded in GOAL_NAVIGATION",
    )

    declare_nav2_success_min_wait_sec = DeclareLaunchArgument(
        "nav2_success_min_wait_sec",
        default_value="1.0",
        description="Minimum seconds after publishing a Nav2 goal before accepting a success status",
    )

    declare_llm_candidate_selection_enabled = DeclareLaunchArgument(
        "llm_candidate_selection_enabled",
        default_value="true",
        description="Use the Qwen/Ollama model to choose the best view after deterministic scoring",
    )

    declare_llm_model = DeclareLaunchArgument(
        "llm_model",
        default_value="qwen2.5:7b",
        description="Ollama model used for graph_nav view selection",
    )

    declare_ollama_base_url = DeclareLaunchArgument(
        "ollama_base_url",
        default_value="http://127.0.0.1:11434",
        description="Base URL for the Ollama HTTP API",
    )

    declare_llm_request_timeout_sec = DeclareLaunchArgument(
        "llm_request_timeout_sec",
        default_value="60.0",
        description="Seconds to wait for the Qwen/Ollama view-selection response",
    )

    declare_llm_candidate_num_predict = DeclareLaunchArgument(
        "llm_candidate_num_predict",
        default_value="700",
        description="Maximum generated tokens for the Qwen/Ollama view-selection response",
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
        condition=IfCondition(use_realsense),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch),
        condition=IfCondition(use_slam),
    )
    
    clipseg_debug = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(clipseg_debug_launch),
        condition=IfCondition(use_clipseg),
    )

    # # When mounted on frame
    # base_to_camera = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="base_to_camera_tf",
    #     arguments=[
    #         "0.125", "-0.15", "0.825",
    #         "0", "0", "0",
    #         "panther/base_link",
    #         "camera_link",
    #     ],
    #     output="screen",
    # )

    # When mounted on the LeRobot arm
    # base_to_camera = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="base_to_camera_tf",
    #     arguments=[
    #         "-0.125", "0.02", "0.818",
    #         "0", "0", "0",
    #         "panther/base_link",
    #         "camera_link",
    #     ],
    #     output="screen",
    # )

    # camera_to_depth_optical = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="camera_to_depth_optical_tf",
    #     arguments=[
    #         "0", "0", "0",
    #         "-1.57079632679", "0", "-1.57079632679",
    #         "camera_link",
    #         "camera_depth_optical_frame",
    #     ],
    #     output="screen",
    # )

    base_to_velodyne = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_velodyne_tf",
        condition=IfCondition(use_lidar),
        arguments=[
            "0.125", "0.02", "0.643",
            "0", "0", "0",
            "panther/base_link",
            "panther/velodyne_link",
        ],
        output="screen",
    )

    task_memory = OpaqueFunction(
        function=store_task_prompt,
        condition=IfCondition(use_task_memory),
    )

    lerobot_server = ExecuteProcess(
        cmd=[lerobot_server_python, lerobot_server_script],
        output="screen",
        condition=IfCondition(use_lerobot_server),
    )

    # GUI
    # logger = Node(
    #     package="camera_capture",
    #     executable="lerobot_realsense_sequence_logger",
    #     name="lerobot_realsense_sequence_logger",
    #     output="screen",
    #     parameters=[{
    #         "nodes_dir": nodes_dir,
    #         "global_frame": "map",
    #         "robot_frame": "panther/base_link",

    #         "lerobot_server_url": lerobot_server_url,

    #         # 8 semantic camera stops. The logger sends these as stop_index/stop_name
    #         # to the LeRobot HTTP server. The server maps them to raw m5 positions.
    #         "camera_stop_names": [
    #             "back",
    #             "back_left",
    #             "left",
    #             "front_left",
    #             "front",
    #             "front_right",
    #             "right",
    #             "back_right",
    #         ],

    #         "settle_after_motion_sec": 0.5,
    #         "wait_for_fresh_frame_sec": 60.0,
    #         "fresh_frames_to_skip": 0,

    #         "realsense_color_topic": realsense_color_topic,
    #         "realsense_aligned_depth_topic": realsense_depth_topic,
    #         "require_all_topics": True,
    #     }],
    #     condition=IfCondition(use_logger),
    # )
    
    clipseg = Node(
        package="robot_navigation",
        executable="clipseg_depth_node_with_mask",
        name="clipseg_depth_node",
        output="screen",
        condition=IfCondition(use_clipseg),
        parameters=[
            {
                "rgb_topic": "/camera/color/image_compressed",
                "depth_topic": "/camera/depth/image_compressed",
                "mask_threshold": mask_threshold,
            }
        ],
        arguments=[
            "--ros-args",
            "-p",
            ["prompts:=['", target_prompt, "']"],
        ],
    )

    masked_depth_target_point = Node(
        package="robot_navigation",
        executable="masked_depth_target_point",
        name="masked_depth_target_point",
        output="screen",
        condition=IfCondition(use_clipseg),
        parameters=[
            {
                "masked_depth_topic": "/clipseg/masked_depth",
                "camera_info_topic": "/camera/depth/camera_info_compressed",
                "target_point_topic": target_point_topic,
                "target_frame": "map",
                "min_valid_pixels": masked_depth_min_valid_pixels,
                "min_depth_m": masked_depth_min_depth_m,
                "max_depth_m": masked_depth_max_depth_m,
            }
        ],
    )

    rgbd2pointcloud = Node(
        package="robot_navigation",
        executable="rgbd2pointcloud",
        name="rgbd2pointcloud",
        output="screen",
        condition=IfCondition(use_clipseg),
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
    
    lerobot_camera_tf = Node(
        package="camera_capture",
        executable="lerobot_camera_tf_broadcaster",
        name="lerobot_camera_tf_broadcaster",
        output="screen",
        condition=IfCondition(use_lerobot_camera_tf),
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

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch),
        condition=IfCondition(use_nav2),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_graph_nav",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
    )

    graph_nav_state_machine = Node(
        package="graph_nav",
        executable="graph_nav_state_machine3",
        name="graph_nav_state_machine",
        output="screen",
        parameters=[{
            "user_prompt": user_prompt,
            "target_prompt": target_prompt,
            "nodes_dir": nodes_dir,
            "state_file_name": graph_nav_state_file_name,
            "phase_topic": graph_nav_phase_topic,
            "finished_topic": graph_nav_finished_topic,
            "target_visible_topic": graph_nav_target_visible_topic,
            "goal_reached_topic": graph_nav_goal_reached_topic,
            "node_reached_topic": graph_nav_node_reached_topic,
            "target_point_topic": target_point_topic,
            "target_visible_from_target_point": target_visible_from_target_point,
            "target_visible_min_observations": target_visible_min_observations,
            "target_visible_window_sec": target_visible_window_sec,
            "target_visible_min_range_m": target_visible_min_range_m,
            "target_visible_max_range_m": target_visible_max_range_m,
            "target_visible_max_spread_m": target_visible_max_spread_m,
            "stop_scan_on_target_visible": stop_scan_on_target_visible,
            "cmd_vel_topic": cmd_vel_topic,
            "tick_sec": graph_nav_tick_sec,
            "global_frame": "map",
            "robot_frame": "panther/base_link",
            "tf_timeout_sec": 1.0,
            "lerobot_server_url": lerobot_server_url,
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
            "settle_after_motion_sec": settle_after_motion_sec,
            "settle_before_sweep_sec": settle_before_sweep_sec,
            "wait_for_fresh_frame_sec": 60.0,
            "fresh_frames_to_skip": fresh_frames_to_skip,
            "wait_for_fresh_color_frame_sec": wait_for_fresh_color_frame_sec,
            "realsense_color_topic": realsense_color_topic,
            "realsense_aligned_depth_topic": realsense_depth_topic,
            "realsense_depth_camera_info_topic": realsense_depth_camera_info_topic,
            "depth_save_formats": ["png", "npz", "preview_png"],
            "max_depth_preview_m": max_depth_preview_m,
            "moondream_server_url": moondream_server_url,
            "require_moondream_descriptions": require_moondream_descriptions,
            "moondream_overwrite_descriptions": moondream_overwrite_descriptions,
            "candidate_top_k": candidate_top_k,
            "candidate_memory_nodes": candidate_memory_nodes,
            "next_node_goal_topic": next_node_goal_topic,
            "nav2_goal_pose_topic": nav2_goal_pose_topic,
            "publish_nav2_goal_pose": publish_nav2_goal_pose,
            "next_node_goal_distance_m": next_node_goal_distance_m,
            "nav2_status_topic": nav2_status_topic,
            "auto_scan_after_nav2_success": auto_scan_after_nav2_success,
            "auto_finish_on_goal_nav2_success": auto_finish_on_goal_nav2_success,
            "nav2_success_min_wait_sec": nav2_success_min_wait_sec,
            "llm_candidate_selection_enabled": llm_candidate_selection_enabled,
            "llm_model": llm_model,
            "ollama_base_url": ollama_base_url,
            "llm_request_timeout_sec": llm_request_timeout_sec,
            "llm_candidate_num_predict": llm_candidate_num_predict,
            "auto_scan_initial_node": True,
        }],
        condition=IfCondition(use_graph_nav_state_machine),
    )

    return LaunchDescription([
        declare_use_lidar,
        declare_use_realsense,
        declare_use_slam,
        declare_use_nav2,
        declare_use_clipseg,
        declare_use_rviz,
        declare_use_graph_nav_state_machine,
        declare_use_lerobot_server,
        declare_use_lerobot_camera_tf,
        declare_use_task_memory,
        declare_user_prompt,
        declare_target_prompt,
        declare_mask_threshold,
        declare_use_logger,
        declare_nodes_dir,
        declare_task_file_name,
        declare_lerobot_server_url,
        declare_lerobot_server_python,
        declare_lerobot_server_script,
        declare_camera_tf_source,
        declare_lerobot_state_path,
        declare_lerobot_yaw_path,
        declare_lerobot_position_path,
        declare_lerobot_position_zero,
        declare_lerobot_position_to_rad,
        declare_lerobot_joint_state_topic,
        declare_lerobot_yaw_joint_name,
        declare_realsense_color_topic,
        declare_realsense_depth_topic,
        declare_realsense_depth_camera_info_topic,
        declare_max_depth_preview_m,
        declare_moondream_server_url,
        declare_require_moondream_descriptions,
        declare_moondream_overwrite_descriptions,
        declare_rviz_config,
        declare_graph_nav_state_file_name,
        declare_graph_nav_phase_topic,
        declare_graph_nav_finished_topic,
        declare_graph_nav_target_visible_topic,
        declare_graph_nav_goal_reached_topic,
        declare_graph_nav_node_reached_topic,
        declare_target_point_topic,
        declare_target_visible_from_target_point,
        declare_target_visible_min_observations,
        declare_target_visible_window_sec,
        declare_target_visible_min_range_m,
        declare_target_visible_max_range_m,
        declare_target_visible_max_spread_m,
        declare_stop_scan_on_target_visible,
        declare_masked_depth_min_valid_pixels,
        declare_masked_depth_min_depth_m,
        declare_masked_depth_max_depth_m,
        declare_cmd_vel_topic,
        declare_graph_nav_tick_sec,
        declare_settle_after_motion_sec,
        declare_settle_before_sweep_sec,
        declare_fresh_frames_to_skip,
        declare_wait_for_fresh_color_frame_sec,
        declare_candidate_top_k,
        declare_candidate_memory_nodes,
        declare_next_node_goal_topic,
        declare_nav2_goal_pose_topic,
        declare_publish_nav2_goal_pose,
        declare_next_node_goal_distance_m,
        declare_nav2_status_topic,
        declare_auto_scan_after_nav2_success,
        declare_auto_finish_on_goal_nav2_success,
        declare_nav2_success_min_wait_sec,
        declare_llm_candidate_selection_enabled,
        declare_llm_model,
        declare_ollama_base_url,
        declare_llm_request_timeout_sec,
        declare_llm_candidate_num_predict,

        TimerAction(period=0.2, actions=[
            task_memory,
        ]),

        TimerAction(period=0.5, actions=[
            lerobot_server,
        ]),

        # Static TFs first.
        TimerAction(period=0.5, actions=[
            # base_to_camera,
            # camera_to_depth_optical,
            base_to_velodyne,
        ]),

        TimerAction(period=1.0, actions=[
            lerobot_camera_tf,
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
            masked_depth_target_point,
        ]),

        TimerAction(period=10.0, actions=[
            rgbd2pointcloud,
        ]),

        TimerAction(period=11.0, actions=[
            crop_lidar,
        ]),

        # Nav2 last, after SLAM has had time to publish map -> odom.
        TimerAction(period=12.0, actions=[
            nav2,
        ]),
        
        # TimerAction(period=13.0, actions=[
        #     logger,
        # ]),

        TimerAction(period=14.0, actions=[
            rviz,
        ]),

        TimerAction(period=15.0, actions=[
            graph_nav_state_machine,
        ]),
    ])
