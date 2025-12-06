import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import  LaunchConfiguration, PathJoinSubstitution, TextSubstitution


def generate_launch_description():

    world_arg = DeclareLaunchArgument(
        'world', default_value='world.sdf',
        description='Name of the Gazebo world file to load'
    )

    pkg_diablo_env_ros = get_package_share_directory('diablo_env_ros')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Add your own gazebo library path here
    gazebo_models_path = "/home/david/gazebo_models"
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    # set_headless_env = [
    #     SetEnvironmentVariable(name='GZ_SIM_HEADLESS', value='1'),
    #     SetEnvironmentVariable(name='GZ_SIM_RENDER_ENGINE', value='ogre2'),
    #     SetEnvironmentVariable(name='DISPLAY', value=':0'),
    #     # SetEnvironmentVariable(name='LIBGL_ALWAYS_SOFTWARE', value='1'),
    # ]

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={
            'gz_args': [
                PathJoinSubstitution([
                    pkg_diablo_env_ros,
                    'worlds',
                    LaunchConfiguration('world')
                ]),
                # TextSubstitution(text=' -s -r')
                TextSubstitution(text=' -r -v -v1')
            ],
            'on_exit_shutdown': 'true'
        }.items()
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(world_arg)

    # for env_var in set_headless_env:
    #     launchDescriptionObject.add_action(env_var)

    launchDescriptionObject.add_action(gazebo_launch)

    return launchDescriptionObject