import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction, EmitEvent
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('line_follower')
    resource = os.path.join(share, 'models')
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('verbosity', default_value='3'),
        DeclareLaunchArgument('render_engine', default_value='ogre2'),
        DeclareLaunchArgument('max_speed', default_value='0.18'),
        DeclareLaunchArgument('straight_speed', default_value='0.45'),
        DeclareLaunchArgument('reference_controller', default_value='true'),
        DeclareLaunchArgument('capture_dir', default_value=''),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource + os.pathsep + existing),
        ExecuteProcess(cmd=['gz', 'sim', '-v', LaunchConfiguration('verbosity'),
                            '--render-engine', LaunchConfiguration('render_engine'), '-r', os.path.join(share, 'worlds', 'line_track.sdf')],
                       condition=UnlessCondition(LaunchConfiguration('headless')), output='screen',
                       on_exit=[EmitEvent(event=Shutdown(reason='Gazebo exited; stopping follower and bridges.'))]),
        ExecuteProcess(cmd=['gz', 'sim', '-v', LaunchConfiguration('verbosity'),
                            '--render-engine', LaunchConfiguration('render_engine'), '-r', '-s', '--headless-rendering',
                            os.path.join(share, 'worlds', 'line_track.sdf')],
                       condition=IfCondition(LaunchConfiguration('headless')), output='screen',
                       on_exit=[EmitEvent(event=Shutdown(reason='Gazebo exited; stopping follower and bridges.'))]),
        Node(package='ros_gz_image', executable='image_bridge',
             arguments=['/camera/image_raw'], output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=['/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                        '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry'], output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='line_imu_bridge',
             arguments=['/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'], output='screen'),
        Node(package='line_follower', executable='reference_follower', output='screen',
             condition=IfCondition(LaunchConfiguration('reference_controller')),
             parameters=[{'straight_speed': ParameterValue(LaunchConfiguration('straight_speed'), value_type=float),
                          'capture_dir': LaunchConfiguration('capture_dir')}]),
        TimerAction(period=20.0, actions=[
            Node(package='line_follower', executable='follower', output='screen',
                 condition=UnlessCondition(LaunchConfiguration('reference_controller')),
                 parameters=[{'max_speed': ParameterValue(LaunchConfiguration('max_speed'), value_type=float),
                              'straight_speed': ParameterValue(LaunchConfiguration('straight_speed'), value_type=float),
                              'capture_dir': LaunchConfiguration('capture_dir')}])]),
    ])
