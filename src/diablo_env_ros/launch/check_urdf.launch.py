import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_diablo_env_ros = get_package_share_directory('diablo_env_ros')

    gazebo_models_path, ignore_last_dir = os.path.split(pkg_diablo_env_ros)
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz.'
    )

    world_arg = DeclareLaunchArgument(
        'world', default_value='world.sdf',
        description='Name of the Gazebo world file to load'
    )

    model_arg = DeclareLaunchArgument(
        'model', default_value='robot.urdf',
        description='Name of the URDF description to load'
    )

    # Define the path to your URDF or Xacro file
    urdf_file_path = PathJoinSubstitution([
        pkg_diablo_env_ros,  # Replace with your package name
        "urdf",
        LaunchConfiguration('model')  # Replace with your URDF or Xacro file
    ])

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_diablo_env_ros, 'launch', 'world.launch.py'),
        ),
        launch_arguments={
        'world': LaunchConfiguration('world'),
        }.items()
    )

    


    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "robot",   
            "-topic", "robot_description",
            "-x", "0.0", "-y", "0.0", "-z", "0.0", "-Y", "0.0",  # Initial spawn position
            # "<gravity>false</gravity>"
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': Command(['xacro', ' ', urdf_file_path]),
             'use_sim_time': True},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )


    # Launch rviz
    
    # Node to bridge messages like /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/effort_cmd@std_msgs/msg/Float64@gz.msgs.Double",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_diablo_env_ros, 'rviz', 'rviz.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': True},
        ]
    )
    
    

    # joint_state_publisher_gui_node = Node(
    #     package='joint_state_publisher_gui',
    #     executable='joint_state_publisher_gui',
    # )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(model_arg)

    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(spawn_urdf_node)
    launchDescriptionObject.add_action(robot_state_publisher_node)
    
    
    # Add controllers with delays
    
    
    # Spawn robot after controllers are ready


    
    launchDescriptionObject.add_action(gz_bridge_node)
     
    launchDescriptionObject.add_action(rviz_node)
    # launchDescriptionObject.add_action(TimerAction(actions=[spawn_urdf_node],period=5.0))
    # launchDescriptionObject.add_action(TimerAction(actions=[robot_state_publisher_node],period=5.0))
    
    
    # # Add controllers with delays
    
    
    # # Spawn robot after controllers are ready


    
    # launchDescriptionObject.add_action(TimerAction(actions=[gz_bridge_node],period=5.0))
     
    # launchDescriptionObject.add_action(TimerAction(actions=[rviz_node],period=5.0))
    # launchDescriptionObject.add_action(joint_state_publisher_gui_node)

    return launchDescriptionObject