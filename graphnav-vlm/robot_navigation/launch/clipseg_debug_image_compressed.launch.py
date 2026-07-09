from launch import LaunchDescription
from launch_ros.actions import Node
import ament_index_python
import os


def generate_launch_description():
    clipseg_compressed_node = Node(
        package='image_transport',
        executable='republish',
        name='republish_clipseg_compressed',
        output='screen',
        arguments=[
            'raw',
            'compressed',
            '--ros-args',
            '-r', 'in:=/clipseg/debug_image',
            '-r', 'out/compressed:=/clipseg/debug_image/compressed'
        ]
    )
    return LaunchDescription([
        clipseg_compressed_node,
    ])