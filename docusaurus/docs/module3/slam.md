# SLAM Implementation

## Overview

Simultaneous Localization and Mapping (SLAM) is a fundamental capability in robotics that allows a robot to build a map of an unknown environment while simultaneously localizing itself within that map. This is essential for autonomous navigation in previously unexplored spaces.

## SLAM Algorithm Explanations

### What is SLAM?

SLAM is the computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it. The challenge lies in the circular dependency: to map the environment you need to know where you are, but to know where you are you need a map of the environment.

### Common SLAM Approaches

#### 1. Graph-Based SLAM

Graph-based SLAM represents the problem as a graph optimization problem where:
- Nodes represent robot poses
- Edges represent constraints between poses
- The goal is to find the maximum likelihood estimate of the robot trajectory and landmark positions

#### 2. Extended Kalman Filter (EKF) SLAM

EKF SLAM maintains a state vector containing both robot pose and landmark positions, along with a covariance matrix representing uncertainty.

#### 3. Particle Filter SLAM (Monte Carlo Localization)

Particle filter approaches maintain multiple hypotheses about the robot's location and map, which is particularly effective for handling ambiguity.

## ROS 2 Navigation Stack Configuration

### Overview of ROS 2 Navigation Stack

The ROS 2 Navigation Stack (Nav2) provides a complete navigation solution including:
- Global and local planners
- Costmap management
- Localization (AMCL)
- Behavior trees for complex behaviors
- Recovery behaviors

### SLAM Toolbox Integration

The SLAM Toolbox is the recommended SLAM solution for ROS 2, providing:
- Online and offline SLAM capabilities
- Support for 2D and 3D mapping
- Real-time and batch processing modes

### Basic SLAM Launch Configuration

Create a launch file for SLAM: `slam_launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params_file = LaunchConfiguration('slam_params_file')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    slam_params_file_arg = DeclareLaunchArgument(
        'slam_params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_slam'),
            'config',
            'slam_params.yaml'
        ]),
        description='Full path to the slam toolbox parameters file'
    )

    # SLAM Toolbox node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[
            slam_params_file,
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        slam_params_file_arg,
        slam_toolbox_node
    ])
```

### SLAM Parameters Configuration

Create a configuration file `config/slam_params.yaml`:

```yaml
slam_toolbox:
  ros__parameters:
    # Plugin configuration
    solver_plugin: solver_plugins::CeresSolver
    ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
    ceres_preconditioner: SCHUR_JACOBI
    ceres_trust_strategy: LEVENBERG_MARQUARDT
    ceres_dogleg_type: TRADITIONAL_DOGLEG
    max_iterations: 500

    # Map parameters
    map_update_interval: 2.0
    resolution: 0.05
    max_laser_range: 20.0
    minimum_time_interval: 0.5
    transform_publish_period: 0.02

    # Trajectory parameters
    debug_logging: false
    throttle_scans: 1
    stack_size_to_use: 40000000  # 40MB
    enable_interactive_mode: true

    # Optimization parameters
    use_scan_matching: true
    use_scan_barycenter: true
    minimum_travel_distance: 0.5
    minimum_travel_heading: 0.5
    scan_buffer_size: 10
    scan_buffer_maximum_scan_distance: 10.0
    link_match_minimum_response_fine: 0.1
    loop_search_maximum_distance: 3.0
    do_loop_closing: true
    loop_match_minimum_chain_size: 10
    loop_match_maximum_variance_coarse: 3.0
    loop_match_minimum_response_coarse: 0.35
    loop_match_minimum_response_fine: 0.40

    # Correspondence parameters
    correspondence_difference: 5
    queue_size: 10

    # Map saving parameters
    save_map_timeout: 10
    first_map_only: false

    # Transform tolerance
    transform_timeout: 0.2
    tf_buffer_duration: 30.

    # Smoothing parameters
    smoothing_alpha: 0.0
```

## Practical SLAM Examples

### Basic SLAM Node Implementation

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

class SimpleSLAMNode(Node):
    def __init__(self):
        super().__init__('simple_slam')

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Publications
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map', 10)

        # SLAM state
        self.current_pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.map_resolution = 0.05  # meters per cell
        self.map_width = 400  # cells
        self.map_height = 400  # cells
        self.map_origin_x = -10.0  # meters
        self.map_origin_y = -10.0  # meters

        # Initialize occupancy grid
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        self.occupancy_grid.fill(-1)  # Unknown

        # Timer for map publishing
        self.map_timer = self.create_timer(1.0, self.publish_map)

        self.get_logger().info('Simple SLAM Node initialized')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        pose = msg.pose.pose
        self.current_pose[0] = pose.position.x
        self.current_pose[1] = pose.position.y

        # Convert quaternion to euler
        quat = pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_pose[2] = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        """Process laser scan for mapping"""
        # Convert robot pose to map coordinates
        robot_map_x = int((self.current_pose[0] - self.map_origin_x) / self.map_resolution)
        robot_map_y = int((self.current_pose[1] - self.map_origin_y) / self.map_resolution)

        # Process each laser beam
        angle = msg.angle_min
        for i, range_val in enumerate(msg.ranges):
            if not (math.isnan(range_val) or range_val > msg.range_max or range_val < msg.range_min):
                # Calculate endpoint of laser beam in world coordinates
                beam_angle = self.current_pose[2] + angle
                end_x = self.current_pose[0] + range_val * math.cos(beam_angle)
                end_y = self.current_pose[1] + range_val * math.sin(beam_angle)

                # Convert to map coordinates
                end_map_x = int((end_x - self.map_origin_x) / self.map_resolution)
                end_map_y = int((end_y - self.map_origin_y) / self.map_resolution)

                # Ray tracing to update map
                self.ray_trace(robot_map_x, robot_map_y, end_map_x, end_map_y)

            angle += msg.angle_increment

    def ray_trace(self, start_x, start_y, end_x, end_y):
        """Ray trace from robot to obstacle"""
        # Bresenham's line algorithm to trace ray
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        x_step = 1 if end_x > start_x else -1
        y_step = 1 if end_y > start_y else -1
        error = dx - dy

        x, y = start_x, start_y

        # Mark free space along the ray
        while x != end_x or y != end_y:
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                # Mark as free space (0)
                self.occupancy_grid[y, x] = 0

            if x == end_x and y == end_y:
                break

            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_step
            if error2 < dx:
                error += dx
                y += y_step

        # Mark endpoint as occupied (100)
        if 0 <= end_x < self.map_width and 0 <= end_y < self.map_height:
            self.occupancy_grid[end_y, end_x] = 100

    def publish_map(self):
        """Publish the occupancy grid map"""
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'

        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Flatten the 2D array to 1D for the message
        map_data = self.occupancy_grid.flatten().tolist()
        map_msg.data = map_data

        self.map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    simple_slam = SimpleSLAMNode()

    try:
        rclpy.spin(simple_slam)
    except KeyboardInterrupt:
        pass
    finally:
        simple_slam.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Using SLAM Toolbox with Real Robot

For a more sophisticated approach using the SLAM Toolbox with a real robot:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class SLAMController(Node):
    def __init__(self):
        super().__init__('slam_controller')

        # Create subscriber for laser scan
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Exploration state
        self.exploration_state = 'forward'  # forward, turn, mapping
        self.safe_distance = 0.8  # meters
        self.exploration_speed = 0.2  # m/s

        # Timer for exploration
        self.explore_timer = self.create_timer(0.1, self.explore_callback)

        self.get_logger().info('SLAM Controller initialized')

    def scan_callback(self, msg):
        """Process laser scan for exploration decisions"""
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
                self.exploration_state = 'turn'
            else:
                self.exploration_state = 'forward'

    def explore_callback(self):
        """Control robot for exploration during SLAM"""
        cmd_msg = Twist()

        if self.exploration_state == 'forward':
            cmd_msg.linear.x = self.exploration_speed
            cmd_msg.angular.z = 0.0
        elif self.exploration_state == 'turn':
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5  # Turn right

        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    slam_controller = SLAMController()

    try:
        rclpy.spin(slam_controller)
    except KeyboardInterrupt:
        # Stop robot before shutting down
        stop_msg = Twist()
        slam_controller.cmd_vel_pub.publish(stop_msg)
        pass
    finally:
        slam_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### SLAM with Costmap Integration

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import numpy as np
from threading import Lock

class SLAMWithCostmap(Node):
    def __init__(self):
        super().__init__('slam_with_costmap')

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Publications
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/global_costmap/costmap', 10)

        # SLAM parameters
        self.map_resolution = 0.05
        self.map_width = 800
        self.map_height = 800
        self.map_origin_x = -20.0
        self.map_origin_y = -20.0

        # Initialize maps
        self.map_lock = Lock()
        self.occupancy_map = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        self.costmap = np.zeros((self.map_height, self.map_width), dtype=np.int8)

        # Timer for publishing maps
        self.publish_timer = self.create_timer(0.5, self.publish_maps)

        self.get_logger().info('SLAM with Costmap initialized')

    def scan_callback(self, msg):
        """Process laser scan and update maps"""
        with self.map_lock:
            # In a real implementation, you would integrate the scan into the map
            # using proper SLAM algorithms. This is a simplified example.

            # For demonstration, we'll just update the costmap based on laser data
            self.update_costmap_from_scan(msg)

    def update_costmap_from_scan(self, scan_msg):
        """Update costmap based on laser scan"""
        # Process each range reading
        angle = scan_msg.angle_min
        for i, range_val in enumerate(scan_msg.ranges):
            if not (math.isnan(range_val) or range_val > scan_msg.range_max or range_val < scan_msg.range_min):
                # Calculate position of obstacle
                world_x = range_val * math.cos(angle)
                world_y = range_val * math.sin(angle)

                # Convert to map coordinates
                map_x = int((world_x - self.map_origin_x) / self.map_resolution)
                map_y = int((world_y - self.map_origin_y) / self.map_resolution)

                # Mark as obstacle in costmap if within bounds
                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    # Apply obstacle inflation to costmap
                    self.inflate_obstacle(map_x, map_y, inflation_radius=5)

            angle += scan_msg.angle_increment

    def inflate_obstacle(self, center_x, center_y, inflation_radius):
        """Inflate obstacle in costmap"""
        for dx in range(-inflation_radius, inflation_radius + 1):
            for dy in range(-inflation_radius, inflation_radius + 1):
                x = center_x + dx
                y = center_y + dy

                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= inflation_radius:
                        # Calculate cost based on distance (closer = higher cost)
                        cost = int(100 * (1 - distance / inflation_radius))
                        self.costmap[y, x] = max(self.costmap[y, x], cost)

    def publish_maps(self):
        """Publish both occupancy map and costmap"""
        with self.map_lock:
            # Publish occupancy map
            occupancy_msg = self.create_map_message(self.occupancy_map, 'map')
            self.map_pub.publish(occupancy_msg)

            # Publish costmap
            costmap_msg = self.create_map_message(self.costmap, 'map')
            self.costmap_pub.publish(costmap_msg)

    def create_map_message(self, map_data, frame_id):
        """Create OccupancyGrid message from numpy array"""
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = frame_id

        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Flatten the 2D array to 1D
        map_msg.data = map_data.flatten().tolist()

        return map_msg

def main(args=None):
    rclpy.init(args=args)
    slam_costmap = SLAMWithCostmap()

    try:
        rclpy.spin(slam_costmap)
    except KeyboardInterrupt:
        pass
    finally:
        slam_costmap.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## SLAM Best Practices

### For Successful SLAM Implementation

1. **Sensor Quality**: Use high-quality sensors with appropriate range and resolution
2. **Motion Models**: Accurate odometry is crucial for good SLAM performance
3. **Parameter Tuning**: Carefully tune SLAM parameters for your specific environment
4. **Computational Resources**: Ensure sufficient CPU/GPU resources for real-time processing
5. **Environmental Features**: Environments with distinctive features work better for SLAM

### Troubleshooting Common SLAM Issues

1. **Drift**: Caused by accumulated odometry errors; use loop closure to correct
2. **Poor Loop Closure**: Ensure sufficient overlap between map sections
3. **Dynamic Objects**: Filter out dynamic objects that cause false correspondences
4. **Map Quality**: Use appropriate resolution and inflation parameters

## Acceptance Criteria Met

- [X] SLAM algorithm explanations
- [X] ROS 2 Navigation Stack configuration
- [X] Practical SLAM examples