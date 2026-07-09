import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode

import isaac_ros_launch_utils as lu

from nvblox_ros_python_utils.nvblox_constants import NVBLOX_CONTAINER_NAME


def generate_launch_description():
    actions = []

    use_depth = LaunchConfiguration("use_depth")
    use_color = LaunchConfiguration("use_color")
    use_lidar = LaunchConfiguration("use_lidar")
    camera_depth_image = LaunchConfiguration("camera_depth_image")
    camera_depth_info = LaunchConfiguration("camera_depth_info")
    camera_color_image = LaunchConfiguration("camera_color_image")
    camera_color_info = LaunchConfiguration("camera_color_info")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")

    declare_use_depth = DeclareLaunchArgument(
        "use_depth",
        default_value="true",
        description="Enable RealSense depth input for NvBlox",
    )
    declare_use_color = DeclareLaunchArgument(
        "use_color",
        default_value="true",
        description="Enable RealSense color input for NvBlox",
    )
    declare_use_lidar = DeclareLaunchArgument(
        "use_lidar",
        default_value="true",
        description="Enable LiDAR input for NvBlox",
    )
    declare_camera_depth_image = DeclareLaunchArgument(
        "camera_depth_image",
        default_value="/camera0/driver/aligned_depth_to_color/image_raw",
        description="RealSense depth image topic for NvBlox",
    )
    declare_camera_depth_info = DeclareLaunchArgument(
        "camera_depth_info",
        default_value="/camera0/driver/aligned_depth_to_color/camera_info",
        description="RealSense depth camera info topic for NvBlox",
    )
    declare_camera_color_image = DeclareLaunchArgument(
        "camera_color_image",
        default_value="/camera0/driver/color/image_raw",
        description="RealSense color image topic for NvBlox",
    )
    declare_camera_color_info = DeclareLaunchArgument(
        "camera_color_info",
        default_value="/camera0/driver/color/camera_info",
        description="RealSense color camera info topic for NvBlox",
    )
    declare_pointcloud_topic = DeclareLaunchArgument(
        "pointcloud_topic",
        default_value="/velodyne_points",
        description="LiDAR point cloud topic for NvBlox",
    )

    actions.append(lu.set_parameter("use_sim_time", False))

    # NOTE: This launch file assumes sensors and transforms are already running
    # (e.g., from target_pipeline_compressed_nvblox.launch.py)
    # It only starts the NvBlox node for costmap generation.

    actions.extend([
        declare_use_depth,
        declare_use_color,
        declare_use_lidar,
        declare_camera_depth_image,
        declare_camera_depth_info,
        declare_camera_color_image,
        declare_camera_color_info,
        declare_pointcloud_topic,
    ])

    relay_depth_image = Node(
        package="nvblox_costmaps",
        executable="nvblox_depth_relay",
        name="nvblox_depth_relay",
        output="screen",
        parameters=[{
            "input_depth_topic": "/camera0/driver/aligned_depth_to_color/image_raw",
            "input_camera_info_topic": "/camera0/driver/aligned_depth_to_color/camera_info",
            "output_depth_topic": "/nvblox/relay/aligned_depth_to_color/image_raw",
            "output_camera_info_topic": "/nvblox/relay/aligned_depth_to_color/camera_info",
        }],
    )

    actions.append(relay_depth_image)

    # -------------------------
    # Nvblox container
    # -------------------------

    actions.append(
        lu.component_container(
            NVBLOX_CONTAINER_NAME,
            container_type="isolated",
            log_level="info",
        )
    )

    nvblox_base_config = lu.get_path(
        "nvblox_examples_bringup",
        "config/nvblox/nvblox_base.yaml",
    )

    nvblox_node = ComposableNode(
        name="nvblox_node",
        package="nvblox_ros",
        plugin="nvblox::NvbloxNode",
        remappings=[
            # RealSense RGB-D.
            (
                "camera_0/depth/image",
                "/nvblox/relay/aligned_depth_to_color/image_raw",
            ),
            (
                "camera_0/depth/camera_info",
                "/nvblox/relay/aligned_depth_to_color/camera_info",
            ),
            (
                "camera_0/color/image",
                camera_color_image,
            ),
            (
                "camera_0/color/camera_info",
                camera_color_info,
            ),

            # Velodyne.
            (
                "pointcloud",
                pointcloud_topic,
            ),
        ],
        parameters=[
            nvblox_base_config,
            {
                # -------------------------
                # Sensor usage
                # -------------------------
                "num_cameras": 1,

                "use_depth": use_depth,
                "use_color": use_color,
                "use_lidar": use_lidar,

                "use_tf_transforms": True,
                "use_topic_transforms": False,

                # -------------------------
                # Frames
                # -------------------------
                "global_frame": "panther/odom",
                "pose_frame": "panther/base_link",

                "map_clearing_frame_id": "panther/base_link",
                "esdf_slice_bounds_visualization_attachment_frame_id": "panther/base_link",
                "workspace_height_bounds_visualization_attachment_frame_id": "panther/base_link",
                "ground_plane_visualization_attachment_frame_id": "panther/base_link",

                # -------------------------
                # Velodyne VLP-16 model
                # -------------------------
                "lidar_height": 16,
                "lidar_width": 1800,
                "lidar_vertical_fov_rad": 0.523599,

                # VLP-16 is roughly +/- 15 deg.
                "min_angle_below_zero_elevation_rad": 0.261799,
                "max_angle_above_zero_elevation_rad": 0.261799,

                "lidar_min_valid_range_m": 0.5,

                # Disabled because your Velodyne cloud caused crash with motion compensation.
                "use_lidar_motion_compensation": False,
                "pointcloud2_timestamps_are_relative": False,

                # -------------------------
                # Sensor-specific integration range
                # -------------------------

                # RealSense depth integration range.
                # This makes the camera contribute only up to about 3 m.
                # Beyond this, the fused map is effectively LiDAR-only.
                "static_mapper.projective_integrator_max_integration_distance_m": 2.8,

                # Velodyne integration range.
                "static_mapper.lidar_projective_integrator_max_integration_distance_m": 10.0,

                # Back projection / visualization related distance.
                "max_back_projection_distance": 10.0,

                # -------------------------
                # Decay / dynamic-object cleanup
                # -------------------------
                "decay_tsdf_rate_hz": 12.0,
                "static_mapper.tsdf_decay_factor": 0.85,
                "static_mapper.projective_integrator_max_weight": 1.0,
                "static_mapper.raycast_subsampling_factor": 1,

                "static_mapper.tsdf_set_free_distance_on_decayed": True,
                "static_mapper.decay_integrator_deallocate_decayed_blocks": False,

                # -------------------------
                # ESDF update
                # -------------------------
                "update_esdf_rate_hz": 20.0,
                "publish_layer_rate_hz": 10.0,
                "publish_debug_vis_rate_hz": 5.0,

                "static_mapper.esdf_integrator_max_distance_m": 1.5,
                "static_mapper.esdf_integrator_min_weight": 0.02,

                # -------------------------
                # Nav2 ESDF slice height band
                # -------------------------
                # These are in the Nvblox global frame.
                # With base_link around z=0 and floor around z=-0.182,
                # min=-0.05 means ~13 cm above ground.
                "static_mapper.esdf_slice_min_height": 0.3,
                "static_mapper.esdf_slice_max_height": 1.10,
                "static_mapper.esdf_slice_height": 0.05,

                # Keep local map size sane.
                "clear_map_outside_radius_rate_hz": 1.0,
                "map_clearing_radius_m": 10.0,
            },
        ],
    )

    actions.append(
        lu.load_composable_nodes(
            NVBLOX_CONTAINER_NAME,
            [nvblox_node],
        )
    )

    return LaunchDescription(actions)
