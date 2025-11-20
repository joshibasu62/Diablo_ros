import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"),
        launch_arguments=[("gz_args", ["-r -v 4 empty.sdf"])],
    )

    gazebo_spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_cart_pole",
        arguments=["-name", "diablo", "-topic", "robot_description"],
        output="screen",
    )

    xacro_file = os.path.join(get_package_share_directory("diablo_env_ros"), "urdf", "robot.xacro")
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": Command(["xacro ", xacro_file])}, {"use_sim_time": True}],
    )

    bridge_config = os.path.join(get_package_share_directory("diablo_env_ros"), "config", "gazebo_bridge.yaml")
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ign_bridge",
        ros_arguments=["-p", f"config_file:={bridge_config}"],
        output="screen",
    )

    # simulation_control = Node(
    #     package="simulation_control",
    #     executable="simulation_control_node",
    #     parameters=[{"use_sim_time": True}],
    #     output="screen",
    # )

    # simulation_observation = Node(
    #     package="cart_pole_observation",
    #     executable="cart_pole_observation_node",
    #     parameters=[{"use_sim_time": True}],
    #     output="screen",
    # )

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz.'
    )

    # Launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # arguments=['-d', os.path.join(get_package_share_directory("diablo_env_ros"), 'rviz', 'rviz.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': True},
        ]
    )



    return LaunchDescription(
        [gazebo, gazebo_spawn_robot, robot_state_publisher, gz_bridge, rviz_launch_arg, rviz_node]
    )
