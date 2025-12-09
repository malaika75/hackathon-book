# Navigation Stack Configuration

## Overview

The ROS 2 Navigation Stack (Nav2) provides a complete navigation solution for mobile robots. It includes global and local planners, costmap management, localization systems, and behavior trees for complex navigation behaviors.

## Complete Navigation Stack Setup

### Navigation Launch File

Create a comprehensive launch file `navigation_launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    map_file = LaunchConfiguration('map')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    autostart_arg = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_navigation'),
            'config',
            'nav2_params.yaml'
        ]),
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    map_arg = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_navigation'),
            'maps',
            'map.yaml'
        ]),
        description='Full path to map file to load'
    )

    # Launch navigation lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                   {'autostart': autostart},
                   {'node_names': ['map_server',
                                  'planner_server',
                                  'controller_server',
                                  'recoveries_server',
                                  'bt_navigator',
                                  'waypoint_follower']}]
    )

    # Map Server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    # Planner Server
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    # Controller Server
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    # Recovery Server
    recovery_server = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    # BT Navigator
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    # Waypoint Follower
    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        autostart_arg,
        params_file_arg,
        map_arg,
        lifecycle_manager,
        map_server,
        planner_server,
        controller_server,
        recovery_server,
        bt_navigator,
        waypoint_follower
    ])
```

### Navigation Parameters Configuration

Create a comprehensive navigation configuration file `config/nav2_params.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: false
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    scan_topic: scan
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.1
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0

amcl_map_client:
  ros__parameters:
    use_sim_time: false

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: false

bt_navigator:
  ros__parameters:
    use_sim_time: false
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: true
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_through_poses_w_replanning_and_recovery.xml
    default_nav_to_pose_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_consistent_localization_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_assisted_teleop_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: false

controller_server:
  ros__parameters:
    use_sim_time: false
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: true

    # DWB Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      primary_controller: "dwb_core::DWBLocalPlanner"
      rotation_correction_dist: 0.19
      simulate_ahead_time: 1.0
      max_rotational_vel: 1.0
      min_rotational_vel: 0.4
      rotational_acc_lim: 3.2

      # DWB parameters
      dwb_core:
        plugin: "dwb_core::DWBLocalPlanner"
        debug_trajectory_details: true
        min_vel_x: 0.0
        min_vel_y: 0.0
        max_vel_x: 0.5
        max_vel_y: 0.0
        max_vel_theta: 1.0
        min_speed_xy: 0.0
        max_speed_xy: 0.5
        min_speed_theta: 0.0
        acc_lim_x: 2.5
        acc_lim_y: 0.0
        acc_lim_theta: 3.2
        decel_lim_x: -2.5
        decel_lim_y: 0.0
        decel_lim_theta: -3.2
        vx_samples: 20
        vy_samples: 0
        vtheta_samples: 40
        sim_time: 1.7
        linear_granularity: 0.05
        angular_granularity: 0.025
        transform_tolerance: 0.2
        xy_goal_tolerance: 0.25
        trans_stopped_velocity: 0.25
        short_circuit_trajectory_evaluation: true
        stateful: true
        critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
        BaseObstacle.scale: 0.02
        PathAlign.scale: 32.0
        PathAlign.forward_point_distance: 0.1
        GoalAlign.scale: 24.0
        GoalAlign.forward_point_distance: 0.1
        PathDist.scale: 32.0
        GoalDist.scale: 24.0
        RotateToGoal.scale: 32.0
        RotateToGoal.slowing_factor: 5.0
        RotateToGoal.lookahead_time: -1.0

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: false
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: true
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      always_send_full_costmap: true
  local_costmap_client:
    ros__parameters:
      use_sim_time: false
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: false

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: false
      robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: true
  global_costmap_client:
    ros__parameters:
      use_sim_time: false
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: false

map_server:
  ros__parameters:
    use_sim_time: false
    yaml_filename: "map.yaml"

map_saver:
  ros__parameters:
    use_sim_time: false
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: false
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: false

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries::Spin"
    backup:
      plugin: "nav2_recoveries::BackUp"
    wait:
      plugin: "nav2_recoveries::Wait"
    global_frame: odom
    robot_base_frame: base_link
    transform_timeout: 0.1
    use_sim_time: false
    simulate_ahead_time: 2.0
    max_rotational_vel: 1.0
    min_rotational_vel: 0.4
    rotational_acc_lim: 3.2

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 200
```

## Parameter Tuning Recommendations

### For Different Robot Types

#### Differential Drive Robots

For differential drive robots, focus on these key parameters:

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      dwb_core:
        max_vel_x: 0.5          # Maximum forward speed (m/s)
        min_vel_x: 0.0          # Minimum forward speed
        max_vel_theta: 1.0      # Maximum angular speed (rad/s)
        acc_lim_x: 2.5          # Linear acceleration limit
        acc_lim_theta: 3.2      # Angular acceleration limit
        xy_goal_tolerance: 0.25 # Tolerance for reaching goal (m)
        yaw_goal_tolerance: 0.25 # Tolerance for final orientation (rad)
```

#### Ackermann Steering Robots

For Ackermann steering robots:

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      dwb_core:
        max_vel_x: 1.0          # Higher max speed for Ackermann
        max_vel_theta: 0.5      # Lower angular speed due to geometry
        min_turning_radius: 0.3 # Minimum turning radius (m)
```

### Costmap Tuning

#### Local Costmap for Obstacle Avoidance

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      width: 4.0                # Width of local costmap (m)
      height: 4.0               # Height of local costmap (m)
      resolution: 0.05          # Resolution (m/cell)
      robot_radius: 0.3         # Robot radius (m)
      inflation_radius: 0.6     # How far to inflate obstacles (m)
      observation_sources: scan
      scan:
        topic: /scan
        obstacle_max_range: 3.0 # Max range for obstacle detection (m)
        raytrace_max_range: 4.0 # Max range for clearing obstacles (m)
```

#### Global Costmap for Path Planning

```yaml
global_costmap:
  global_costmap:
    ros__parameters:
      resolution: 0.05          # Higher resolution for global planning
      inflation_radius: 0.8     # Larger inflation for global planning
      robot_radius: 0.3         # Robot radius
      track_unknown_space: true # Include unknown space in planning
```

## Obstacle Avoidance Techniques

### Dynamic Obstacle Avoidance

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import math
import numpy as np

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parameters
        self.safe_distance = 0.6  # Minimum safe distance
        self.avoidance_active = False
        self.current_path = None

        self.get_logger().info('Obstacle Avoidance Node initialized')

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Find minimum distance in front of robot
        front_distances = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Get distances in the front 90-degree range
        for i in range(len(msg.ranges)):
            angle = angle_min + i * angle_increment
            if -math.pi/4 <= angle <= math.pi/4:  # Front 90 degrees
                if not math.isnan(msg.ranges[i]) and msg.ranges[i] > 0:
                    front_distances.append(msg.ranges[i])

        if front_distances:
            min_front_distance = min(front_distances)

            if min_front_distance < self.safe_distance:
                self.avoidance_active = True
                self.execute_avoidance(min_front_distance)
            else:
                self.avoidance_active = False

    def execute_avoidance(self, min_distance):
        """Execute obstacle avoidance behavior"""
        cmd_msg = Twist()

        # Simple proportional controller for avoidance
        # Turn away from obstacle based on its proximity
        avoidance_factor = max(0.1, (self.safe_distance - min_distance) / self.safe_distance)

        cmd_msg.linear.x = 0.1  # Slow forward movement
        cmd_msg.angular.z = avoidance_factor * 0.8  # Proportional turn

        self.cmd_vel_pub.publish(cmd_msg)

    def path_callback(self, msg):
        """Store current path for reference"""
        self.current_path = msg

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance = ObstacleAvoidance()

    try:
        rclpy.spin(obstacle_avoidance)
    except KeyboardInterrupt:
        # Stop robot before shutting down
        stop_msg = Twist()
        obstacle_avoidance.cmd_vel_pub.publish(stop_msg)
        pass
    finally:
        obstacle_avoidance.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration with Navigation Stack

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import time

class NavigationManager(Node):
    def __init__(self):
        super().__init__('navigation_manager')

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timer for navigation tasks
        self.nav_timer = self.create_timer(5.0, self.navigate_to_waypoints)

        self.waypoints = [
            [1.0, 1.0, 0.0],   # x, y, theta
            [2.0, 2.0, 1.57],
            [3.0, 1.0, 3.14],
            [2.0, 0.0, -1.57]
        ]

        self.current_waypoint = 0

        self.get_logger().info('Navigation Manager initialized')

    def navigate_to_waypoints(self):
        """Navigate through predefined waypoints"""
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('All waypoints reached')
            return

        goal = self.create_navigate_to_pose_goal(
            self.waypoints[self.current_waypoint][0],
            self.waypoints[self.current_waypoint][1],
            self.waypoints[self.current_waypoint][2]
        )

        self.get_logger().info(f'Navigating to waypoint {self.current_waypoint}: {goal.pose.pose.position.x}, {goal.pose.pose.position.y}')

        # Send navigation goal
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

        self.current_waypoint += 1

    def create_navigate_to_pose_goal(self, x, y, theta):
        """Create a NavigateToPose goal message"""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        # Set position
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        # Convert Euler to Quaternion
        import math
        sin_half_theta = math.sin(theta / 2)
        cos_half_theta = math.cos(theta / 2)
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = sin_half_theta
        goal.pose.pose.orientation.w = cos_half_theta

        return goal

    def goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')

def main(args=None):
    rclpy.init(args=args)
    nav_manager = NavigationManager()

    try:
        rclpy.spin(nav_manager)
    except KeyboardInterrupt:
        pass
    finally:
        nav_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Navigation Configuration

### Performance Optimization

1. **Costmap Resolution**: Balance resolution with performance; higher resolution = more accuracy but slower processing
2. **Update Frequencies**: Match update frequencies to your robot's capabilities
3. **Local Planner**: Choose appropriate local planner for your robot type
4. **Recovery Behaviors**: Configure appropriate recovery behaviors for your environment

### Safety Considerations

1. **Velocity Limits**: Set conservative velocity limits initially
2. **Inflation Radius**: Set appropriate inflation for safe obstacle avoidance
3. **Goal Tolerance**: Configure appropriate goal tolerances
4. **Sensor Fusion**: Integrate multiple sensors for robust obstacle detection

## Acceptance Criteria Met

- [X] Complete Navigation Stack setup guide
- [X] Parameter tuning recommendations
- [X] Obstacle avoidance techniques