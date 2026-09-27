from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='subscriber_to_observation_messages', 
            executable='subscriber', 
            name='diablo_observer',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='simulation_control',    
                    executable='simulation_control_node',
                    name='simulation_control',
                    output='screen',
                    parameters=[{'use_sim_time': True}]
                )
            ]
        ),
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='base_class',             
                    executable='actor_critic_node',
                    name='actor_critic_node',
                    output='screen',
                    parameters=[{'use_sim_time': True}]
                )
            ]
        )
    ])

   