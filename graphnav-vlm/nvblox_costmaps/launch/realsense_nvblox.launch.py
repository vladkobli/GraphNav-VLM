from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # Static TF from robot base to the RealSense optical/body frame.
    # Keep this only if your RealSense launch does NOT already publish this TF.
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

    nvblox_container = ComposableNodeContainer(
        name="nvblox_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                name="nvblox_node",
                package="nvblox_ros",
                plugin="nvblox::NvbloxNode",
                parameters=[
                    {
                        "num_cameras": 1,

                        "global_frame": "panther/odom",
                        "pose_frame": "panther/base_link",

                        "use_tf_transforms": True,
                        "use_topic_transforms": False,

                        "use_depth": True,
                        "use_color": True,
                        "use_lidar": False,

                        "map_clearing_frame_id": "panther/base_link",
                        "esdf_slice_bounds_visualization_attachment_frame_id": "panther/base_link",
                        "workspace_height_bounds_visualization_attachment_frame_id": "panther/base_link",

                        "static_mapper.esdf_slice_min_height": 0.2,
                        "static_mapper.esdf_slice_max_height": 1.10,
                        "static_mapper.esdf_slice_height": 0.825,
                    }
                ],
                remappings=[
                    (
                        "camera_0/depth/image",
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
                ],
            )
        ],
    )

    return LaunchDescription([
        camera_tf,
        nvblox_container,
    ])