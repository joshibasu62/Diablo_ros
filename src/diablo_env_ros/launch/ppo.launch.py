from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():

    subscriber_node = Node(
        package='subscriber_to_observation_messages', 
        executable='subscriber', 
        name='diablo_observer',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    
    simulation_node = Node(
        package='simulation_control',    
        executable='simulation_control_node',
        name='simulation_control',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    
    rl_node = Node(
        package='base_class',             
        executable='train_ppo_diablo',
        name='train_ppo_diablo',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    
    launch_description = LaunchDescription()

    launch_description.add_action(subscriber_node)

    launch_description.add_action(
        TimerAction(
            period=2.0,
            actions=[simulation_node]
        )
    )

    launch_description.add_action(
        TimerAction(
            period=4.0,
            actions=[rl_node]
        )
    )

    return launch_description
