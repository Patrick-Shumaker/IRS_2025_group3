from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    pkg_share   = FindPackageShare('hand_solo_virtual_nav')
    slam_params = PathJoinSubstitution([pkg_share, 'config', 'pa_slam_params.yaml'])
    rviz_cfg    = PathJoinSubstitution([pkg_share, 'rviz',   'pa_rviz_mapping.rviz'])

    return LaunchDescription([
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            parameters=[slam_params, {'use_sim_time': False}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_cfg],
            parameters=[{'use_sim_time': False}]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_base_to_lidar',
            arguments=[
                '0','0','0',            # x y z
                '1','0','0','0',        # qx qy qz qw  (use 0 0 0 1 for identity)
                'virtual_hand_solo/base_link',
                'virtual_hand_solo/lidar_link'
            ]
        ),
    ])
