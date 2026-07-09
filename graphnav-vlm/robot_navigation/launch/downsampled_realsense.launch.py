from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    resize_color_node = Node(
        package='robot_navigation',
        executable='resize_image_node',
        name='resize_color_compressed',
        output='screen',
        parameters=[{
            'input_topic': '/camera0/driver/color/image_raw',
            'output_topic': '/camera/color/image_compressed',
            'width': 352,
            'height': 198,
        }]
    )

    resize_color_info_node = Node(
        package='robot_navigation',
        executable='resize_camera_info_node',
        name='resize_color_camera_info_compressed',
        output='screen',
        parameters=[{
            'input_topic': '/camera0/driver/color/camera_info',
            'output_topic': '/camera/color/camera_info_compressed',
            'new_width': 352,
            'new_height': 198,
        }]
    )

    resize_depth_node = Node(
        package='robot_navigation',
        executable='resize_image_node',
        name='resize_depth_compressed',
        output='screen',
        parameters=[{
            # Prefer this if the topic exists:
            'input_topic': '/camera0/driver/aligned_depth_to_color/image_raw',

            # Fallback if aligned topic does not exist:
            # 'input_topic': '/camera0/driver/depth/image_rect_raw',

            'output_topic': '/camera/depth/image_compressed',
            'width': 352,
            'height': 198,
        }]
    )

    resize_depth_info_node = Node(
        package='robot_navigation',
        executable='resize_camera_info_node',
        name='resize_depth_camera_info_compressed',
        output='screen',
        parameters=[{
            # Match the depth image source above.
            'input_topic': '/camera0/driver/aligned_depth_to_color/camera_info',

            # Fallback if using /camera0/driver/depth/image_rect_raw:
            # 'input_topic': '/camera0/driver/depth/camera_info',

            'output_topic': '/camera/depth/camera_info_compressed',
            'new_width': 352,
            'new_height': 198,
        }]
    )

    return LaunchDescription([
        resize_color_node,
        resize_color_info_node,
        resize_depth_node,
        resize_depth_info_node,
    ])