from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='frontier_exploration',
            executable='classical_frontier_detector',
            name='classical_frontier_detector',
            output='screen',
            parameters=[{
                # Tune this later
                "region_size_thresh": 25,

                # Panther is wider than a TurtleBot.
                # Use the real Panther width + safety margin.
                "robot_width": 0.8,

                # Your slam_toolbox publishes /map
                "occupancy_map_topic": "/map",
            }]
        ),

    ])