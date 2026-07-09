import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, FindExecutable

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_navigation_dir = get_package_share_directory("robot_navigation")

    use_sim_time = LaunchConfiguration("use_sim_time")

    sensors_xacro = os.path.join(
        robot_navigation_dir,
        "urdf",
        "panther_sensors_only.urdf.xacro"
    )

    sensors_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            sensors_xacro,
        ]),
        value_type=str,
    )

    # Publishes ONLY:
    # panther/base_footprint -> panther/velodyne_link
    #
    # The RealSense camera TF is intentionally not published from this
    # sensor-only URDF. When the camera is mounted on the LeRobot pan axis,
    # camera_link must come from the dynamic LeRobot TF broadcaster.
    #
    # It does NOT republish the Panther body/wheels.
    sensors_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="sensors_state_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": sensors_description,
            }
        ],
        remappings=[
            ("/robot_description", "/panther_sensors_description"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),

        sensors_state_publisher,
    ])
