from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Package paths
    pkg_share = FindPackageShare('hand_solo_virtual_nav')
    
    # File paths
    rviz_cfg = PathJoinSubstitution([pkg_share, 'rviz', 'pa_rviz_nav2.rviz'])
    map_yaml = PathJoinSubstitution([pkg_share, 'map', 'pa_warehouse_map_01.yaml'])
    nav2_params = PathJoinSubstitution([pkg_share, 'config', 'pa_nav2_params.yaml'])
    
    # Launch arguments (optional - for flexibility)
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Static transform publisher
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_base_to_lidar',
        arguments=[
            '0', '0', '0',           # x y z
            '1', '0', '0', '0',      # qx qy qz qw 
            'virtual_hand_solo/base_link',
            'virtual_hand_solo/lidar_link'
        ]
    )

    # Map Server - Lifecycle Node
    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'yaml_filename': map_yaml,
             'use_sim_time': use_sim_time}
        ]
    )

    # AMCL - Lifecycle Node
    amcl = LifecycleNode(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time}
        ]
    )

    # Controller Server - Lifecycle Node
    controller_server = LifecycleNode(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time}
        ]
    )

    # Planner Server - Lifecycle Node
    planner_server = LifecycleNode(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time}
        ]
    )

    # Behavior Server - Lifecycle Node
    behavior_server = LifecycleNode(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time}
        ]
    )

    # BT Navigator - Lifecycle Node
    bt_navigator = LifecycleNode(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time}
        ]
    )

    # Waypoint Follower - Lifecycle Node
    waypoint_follower = LifecycleNode(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time}
        ]
    )

    # Velocity Smoother - Lifecycle Node
    velocity_smoother = LifecycleNode(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        namespace='',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', 'cmd_vel')
        ]
    )

    # Lifecycle Manager for Localization
    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['map_server', 'amcl']}
        ]
    )

    # Lifecycle Manager for Navigation
    lifecycle_manager_navigation = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'waypoint_follower',
                'velocity_smoother'
            ]}
        ]
    )

    # RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        # Launch arguments
        declare_use_sim_time,
        
        # TF
        static_tf,
        
        # Localization Stack
        map_server,
        amcl,
        lifecycle_manager_localization,
        
        # Navigation Stack
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        waypoint_follower,
        velocity_smoother,
        lifecycle_manager_navigation,
        
        # Visualization
        rviz
    ])