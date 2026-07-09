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

    # For live hardware, do NOT force use_sim_time=True.
    actions.append(lu.set_parameter('use_sim_time', False))

    # -------------------------
    # Velodyne driver
    # -------------------------
    sensors_share_dir = get_package_share_directory('sensors_bringup')
    driver_params_file = os.path.join(
        sensors_share_dir,
        'config',
        'VLP16-velodyne_driver_node-params.yaml'
    )

    velodyne_driver_node = Node(
        package='velodyne_driver',
        executable='velodyne_driver_node',
        name='velodyne_driver_node',
        output='both',
        parameters=[driver_params_file],
    )

    # -------------------------
    # Velodyne packets -> PointCloud2
    # -------------------------
    velodyne_share_dir = get_package_share_directory('velodyne_pointcloud')
    convert_params_file = os.path.join(
        velodyne_share_dir,
        'config',
        'VLP16-velodyne_transform_node-params.yaml'
    )

    with open(convert_params_file, 'r') as f:
        convert_params = yaml.safe_load(f)['velodyne_transform_node']['ros__parameters']

    convert_params.update({
        'model': 'VLP16',
        'calibration': os.path.join(velodyne_share_dir, 'params', 'VLP16db.yaml'),

        # Important for Nvblox:
        # Nvblox expects LiDAR intrinsics to match the incoming point cloud.
        # For VLP16, try to make /velodyne_points organized: height=16.
        'organize_cloud': True,

        # Keep the cloud in the LiDAR frame.
        # Your driver frame_id is panther/velodyne_link.
        'fixed_frame': 'panther/velodyne_link',
        'target_frame': 'panther/velodyne_link',

        # Reasonable first values.
        'min_range': 0.5,
        'max_range': 50.0,
    })

    velodyne_transform_node = Node(
        package='velodyne_pointcloud',
        executable='velodyne_transform_node',
        name='velodyne_transform_node',
        output='both',
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
            container_type='isolated',
            log_level='info',
        )
    )

    nvblox_base_config = lu.get_path(
        'nvblox_examples_bringup',
        'config/nvblox/nvblox_base.yaml'
    )

    nvblox_node = ComposableNode(
        name='nvblox_node',
        package='nvblox_ros',
        plugin='nvblox::NvbloxNode',
        remappings=[
            # Nvblox subscribes to "pointcloud".
            # Velodyne publishes "/velodyne_points".
            ('pointcloud', '/velodyne_points'),
        ],
        parameters=[
            nvblox_base_config,
            {
                # LiDAR only
                "num_cameras": 0,
                "use_depth": False,
                "use_color": False,
                "use_lidar": True,

                # Keep static TSDF for now, but make it less sticky.
                "mapping_type": "static_tsdf",

                # VLP-16
                "lidar_height": 16,
                "lidar_width": 1800,
                "lidar_vertical_fov_rad": 0.523599,
                "lidar_min_valid_range_m": 0.5,

                "use_lidar_motion_compensation": False,
                "pointcloud2_timestamps_are_relative": False,

                # Range
                "static_mapper.lidar_projective_integrator_max_integration_distance_m": 12.0,
                "max_back_projection_distance": 12.0,
                "map_clearing_radius_m": 12.0,
                "clear_map_outside_radius_rate_hz": 1.0,

                # Update rates
                "integrate_lidar_rate_hz": 40.0,
                "update_esdf_rate_hz": 20.0,
                "publish_layer_rate_hz": 10.0,
                "publish_debug_vis_rate_hz": 5.0,

                # Less sticky, but not so aggressive that far sparse returns vanish immediately
                "decay_tsdf_rate_hz": 12.0,
                "static_mapper.tsdf_decay_factor": 0.80,

                # Keep old obstacles from becoming too confident
                "static_mapper.projective_integrator_max_weight": 1.0,

                # More complete ray clearing
                "static_mapper.raycast_subsampling_factor": 1,

                # When decayed, make it free instead of leaving junk
                "static_mapper.tsdf_set_free_distance_on_decayed": True,
                "static_mapper.decay_integrator_deallocate_decayed_blocks": False,

                # Make ESDF accept lower-weight far returns
                "static_mapper.esdf_integrator_min_weight": 0.03,

                # Slightly thinner occupied band
                "static_mapper.max_tsdf_distance_for_occupancy_m": 0.10,

                # Your height slice
                "static_mapper.esdf_slice_min_height": -0.05,
                "static_mapper.esdf_slice_max_height": 1.00,
                "static_mapper.esdf_slice_height": 0.05,
                
                "global_frame": "panther/odom",
                "pose_frame": "panther/base_link",
                "map_clearing_frame_id": "panther/base_link",

                "esdf_slice_bounds_visualization_attachment_frame_id": "panther/base_link",
                "workspace_height_bounds_visualization_attachment_frame_id": "panther/base_link",
                "ground_plane_visualization_attachment_frame_id": "panther/base_link",
            }
        ],
    )

    actions.append(
        lu.load_composable_nodes(
            NVBLOX_CONTAINER_NAME,
            [nvblox_node],
        )
    )

    # Optional: shutdown everything if the Velodyne driver exits
    actions.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=velodyne_driver_node,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        )
    )

    return LaunchDescription(actions)