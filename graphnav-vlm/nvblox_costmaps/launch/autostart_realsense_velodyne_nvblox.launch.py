import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode

import isaac_ros_launch_utils as lu

from nvblox_ros_python_utils.nvblox_constants import NVBLOX_CONTAINER_NAME


def generate_launch_description():
    actions = []

    actions.append(lu.set_parameter("use_sim_time", False))

    # -------------------------
    # Static TFs
    # -------------------------

    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera0_tf",
        arguments=[
            "0.125", "-0.15", "0.825",
            "0", "0", "0",
            "panther/base_link",
            "camera0_link",
        ],
    )

    velodyne_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_velodyne_tf",
        arguments=[
            "0.125", "0.020", "0.643",
            "0", "0", "0",
            "panther/base_link",
            "panther/velodyne_link",
        ],
    )

    actions.extend([
        camera_tf,
        velodyne_tf,
    ])

    # -------------------------
    # Velodyne driver
    # -------------------------

    sensors_share_dir = get_package_share_directory("sensors_bringup")
    driver_params_file = os.path.join(
        sensors_share_dir,
        "config",
        "VLP16-velodyne_driver_node-params.yaml",
    )

    velodyne_driver_node = Node(
        package="velodyne_driver",
        executable="velodyne_driver_node",
        name="velodyne_driver_node",
        output="both",
        parameters=[driver_params_file],
    )

    # -------------------------
    # Velodyne packets -> PointCloud2
    # -------------------------

    velodyne_share_dir = get_package_share_directory("velodyne_pointcloud")
    convert_params_file = os.path.join(
        velodyne_share_dir,
        "config",
        "VLP16-velodyne_transform_node-params.yaml",
    )

    with open(convert_params_file, "r") as f:
        convert_params = yaml.safe_load(f)["velodyne_transform_node"]["ros__parameters"]

    convert_params.update({
        "model": "VLP16",
        "calibration": os.path.join(velodyne_share_dir, "params", "VLP16db.yaml"),

        # Important for Nvblox LiDAR mode.
        "organize_cloud": True,

        "fixed_frame": "panther/velodyne_link",
        "target_frame": "panther/velodyne_link",

        "min_range": 0.5,
        "max_range": 50.0,
    })

    velodyne_transform_node = Node(
        package="velodyne_pointcloud",
        executable="velodyne_transform_node",
        name="velodyne_transform_node",
        output="both",
        parameters=[convert_params],
    )

    actions.extend([
        velodyne_driver_node,
        velodyne_transform_node,
    ])

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
                # "/camera0/driver/aligned_depth_to_color/image_without_roi",
                "/camera0/driver/aligned_depth_to_color/image_raw",

            ),
            (
                "camera_0/depth/camera_info",
                "/camera0/driver/aligned_depth_to_color/camera_info",
            ),
            (
                "camera_0/color/image",
                "/camera0/driver/color/image_raw",
            ),
            (
                "camera_0/color/camera_info",
                "/camera0/driver/color/camera_info",
            ),

            # Velodyne.
            (
                "pointcloud",
                # "/velodyne_points_without_roi",
                "/velodyne_points",
            ),
        ],
        parameters=[
            nvblox_base_config,
            {
                # -------------------------
                # Sensor usage
                # -------------------------
                "num_cameras": 1,

                "use_depth": True,
                "use_color": True,
                "use_lidar": True,

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

    actions.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=velodyne_driver_node,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        )
    )

    return LaunchDescription(actions)
