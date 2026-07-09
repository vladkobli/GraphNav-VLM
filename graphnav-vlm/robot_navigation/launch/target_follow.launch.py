import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_nav_dir = get_package_share_directory('robot_navigation')
    panther_desc_dir = get_package_share_directory('husarion_ugv_description')

    params_file = os.path.join(robot_nav_dir, 'config', 'nav2_params.yaml')

    urdf = PathJoinSubstitution([
        panther_desc_dir,
        'urdf',
        'panther.urdf.xacro'
    ])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'robot_description': ParameterValue(
                Command([FindExecutable(name='xacro'), ' ', urdf]),
                value_type=str
            )
        }]
    )

    cmd_vel_remaps = [
        ('/cmd_vel', '/panther/cmd_vel'),
        ('cmd_vel', '/panther/cmd_vel'),
    ]

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file],
        remappings=cmd_vel_remaps,
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file],
    )

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[params_file],
        remappings=cmd_vel_remaps,
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file],
        remappings=cmd_vel_remaps,
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
            ],
        }],
    )

    target_to_nav2 = Node(
        package='robot_navigation',
        executable='target_to_nav2',
        name='target_to_nav2',
        output='screen',
        parameters=[{
            'target_topic': '/target_point',
            'global_frame': 'map',
            'robot_frame': 'panther/base_link',
            'stop_distance': 0.5,
            'min_target_distance': 0.35,
            'tf_timeout_sec': 1.0,

            'min_goal_update_period': 3.0,
            'goal_position_epsilon': 1.0,
            'goal_yaw_epsilon_deg': 25.0,
            'goal_reached_holdoff': 1.0,
        }],
    )

    return LaunchDescription([
        robot_state_publisher,
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        lifecycle_manager,
        target_to_nav2,
    ])