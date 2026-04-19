# Chapter 4: Launch Files & Labs

This chapter covers ROS 2 launch files, system orchestration, and includes hands-on labs for basic communication and services/actions.

## ROS 2 Launch Files and System Orchestration

### Introduction

Launch files in ROS 2 allow you to start multiple nodes with specific configurations simultaneously. They provide a powerful way to orchestrate complex robotic systems and manage parameters.

### Python Launch Files

ROS 2 uses Python files with a `.py` extension for launch files. Here's a basic example:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop',
            remappings=[
                ('/turtle1/cmd_vel', '/my_turtle/cmd_vel'),
            ]
        )
    ])
```

### Launch File Structure

A launch file typically includes:

1. **Imports**: Import necessary modules from `launch` and `launch_ros`
2. **Launch Description**: The main function that returns a `LaunchDescription`
3. **Node Actions**: Define the nodes to be launched
4. **Additional Actions**: Parameters, timers, conditions, etc.

### Parameters

Launch files can set parameters for nodes:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    param_file = LaunchConfiguration('param_file')

    declare_param_file_cmd = DeclareLaunchArgument(
        'param_file',
        default_value='/path/to/params.yaml',
        description='Path to parameter file'
    )

    # Node with parameters
    controller_node = Node(
        package='my_package',
        executable='controller_node',
        parameters=[
            param_file,  # Load from YAML file
            {'param1': 'value1'},  # Direct parameter assignment
        ]
    )

    return LaunchDescription([
        declare_param_file_cmd,
        controller_node
    ])
```

### Conditions and Logic

Launch files can include conditional logic:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    # Conditional node launch
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(use_sim_time)
    )

    return LaunchDescription([
        declare_use_sim_time,
        rviz_node
    ])
```

### Parameter Management

Create parameter files in YAML format:

```yaml
# params.yaml
my_robot_controller:
  ros__parameters:
    kp: 1.0
    ki: 0.1
    kd: 0.05
    max_velocity: 1.0
    wheel_radius: 0.05
```

### Best Practices for Launch Files

- Keep launch files in the `launch/` directory of your package
- Use descriptive names for launch files
- Separate different configurations into different launch files
- Use launch arguments to make launch files flexible
- Start dependencies first (e.g., robot state publisher before nodes that need transforms)

---

## Lab 1: ROS 2 Basic Communication

### Objective

In this lab, you will implement publisher/subscriber communication for simple sensor data (e.g., IMU readings) and motor commands using ROS 2.

### Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Basic understanding of ROS 2 concepts (nodes, topics, messages)
- Python or C++ development environment

### Creating the Lab Package

```bash
mkdir -p ~/ros2_labs/src
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab1_communication --dependencies rclpy std_msgs geometry_msgs sensor_msgs
cd ~/ros2_labs
colcon build --packages-select lab1_communication
source install/setup.bash
```

### Task 1: Create IMU Publisher Node

Create the IMU publisher in `lab1_communication/lab1_communication/imu_publisher.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math
import random

class ImuPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')
        self.publisher_ = self.create_publisher(Imu, 'imu_data', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = math.sin(self.i * 0.1) * 0.1
        msg.orientation.w = math.cos(self.i * 0.1) * 0.1

        msg.angular_velocity.x = random.uniform(-0.1, 0.1)
        msg.angular_velocity.y = random.uniform(-0.1, 0.1)
        msg.angular_velocity.z = random.uniform(-0.2, 0.2)

        msg.linear_acceleration.x = random.uniform(-1.0, 1.0)
        msg.linear_acceleration.y = random.uniform(-1.0, 1.0)
        msg.linear_acceleration.z = 9.81 + random.uniform(-0.5, 0.5)

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing IMU data: {msg.linear_acceleration.x:.2f}, {msg.linear_acceleration.y:.2f}, {msg.linear_acceleration.z:.2f}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    imu_publisher = ImuPublisher()

    try:
        rclpy.spin(imu_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        imu_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 2: Create Motor Command Subscriber Node

Create the motor command subscriber in `lab1_communication/lab1_communication/motor_subscriber.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MotorSubscriber(Node):
    def __init__(self):
        super().__init__('motor_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback,
            10)
        self.subscription
        self.get_logger().info('Motor subscriber node initialized')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received motor command - Linear: ({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}), '
                              f'Angular: ({msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f})')
        time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    motor_subscriber = MotorSubscriber()

    try:
        rclpy.spin(motor_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        motor_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 3: Create a Launch File

Create the launch file in `lab1_communication/launch/lab1_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab1_communication',
            executable='imu_publisher',
            name='imu_publisher',
            output='screen'
        ),
        Node(
            package='lab1_communication',
            executable='motor_subscriber',
            name='motor_subscriber',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '/cmd_vel', 'geometry_msgs/msg/Twist',
                 '{linear: {x: 0.5}, angular: {z: 0.2}}', '-r', '1'],
            output='screen'
        )
    ])
```

### Testing Communication

```bash
# Check active topics
ros2 topic list

# Echo IMU data
ros2 topic echo /imu_data

# Send motor commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'
```

---

## Lab 2: ROS 2 Service & Action

### Objective

In this lab, you will create a service for robot state query and an action for a simple navigation task using ROS 2.

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab2_services_actions --dependencies rclpy std_msgs geometry_msgs action_msgs
cd ~/ros2_labs
colcon build --packages-select lab2_services_actions
source install/setup.bash
```

### Task 1: Create Robot State Service

Create the service definition file in `lab2_services_actions/srv/RobotState.srv`:

```
# Request - no parameters needed
---
# Response
string robot_name
float64 battery_level
bool is_charging
float64[] joint_positions
int32 error_code
string error_message
```

Create the service server in `lab2_services_actions/lab2_services_actions/robot_state_server.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from lab2_services_actions.srv import RobotState
import random

class RobotStateService(Node):
    def __init__(self):
        super().__init__('robot_state_service')
        self.srv = self.create_service(RobotState, 'get_robot_state', self.get_robot_state_callback)
        self.get_logger().info('Robot state service server started')

    def get_robot_state_callback(self, request, response):
        response.robot_name = "LabRobot"
        response.battery_level = random.uniform(20.0, 100.0)
        response.is_charging = False
        response.joint_positions = [random.uniform(-1.57, 1.57) for _ in range(6)]
        response.error_code = 0
        response.error_message = "OK"
        self.get_logger().info(f'Service called, returning state for {response.robot_name}')
        return response

def main(args=None):
    rclpy.init(args=args)
    robot_state_service = RobotStateService()

    try:
        rclpy.spin(robot_state_service)
    except KeyboardInterrupt:
        pass
    finally:
        robot_state_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Create the service client in `lab2_services_actions/lab2_services_actions/robot_state_client.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from lab2_services_actions.srv import RobotState

class RobotStateClient(Node):
    def __init__(self):
        super().__init__('robot_state_client')
        self.cli = self.create_client(RobotState, 'get_robot_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = RobotState.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    robot_state_client = RobotStateClient()

    response = robot_state_client.send_request()
    if response:
        robot_state_client.get_logger().info(
            f'Robot: {response.robot_name}\n'
            f'Battery: {response.battery_level:.2f}%\n'
            f'Charging: {response.is_charging}\n'
            f'Joint positions: {[f"{pos:.2f}" for pos in response.joint_positions]}\n'
            f'Error: {response.error_message} (code: {response.error_code})'
        )

    robot_state_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 2: Create Navigation Action

Create the action definition file in `lab2_services_actions/action/MoveToGoal.action`:

```
# Goal
float64 target_distance
float64 max_speed
---
# Result
bool success
float64 actual_distance
string message
---
# Feedback
float64 current_distance
float64 progress_percentage
string status
```

Create the action server in `lab2_services_actions/lab2_services_actions/navigation_action_server.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from lab2_services_actions.action import MoveToGoal
import time

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        self._action_server = ActionServer(
            self,
            MoveToGoal,
            'move_to_goal',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info('Navigation action server started')

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        target_distance = goal_handle.request.target_distance
        max_speed = goal_handle.request.max_speed

        feedback_msg = MoveToGoal.Feedback()
        result = MoveToGoal.Result()

        current_distance = 0.0
        step_size = max_speed * 0.1

        while current_distance < target_distance:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.actual_distance = current_distance
                result.message = 'Goal canceled'
                return result

            current_distance = min(current_distance + step_size, target_distance)
            progress = (current_distance / target_distance) * 100.0

            feedback_msg.current_distance = current_distance
            feedback_msg.progress_percentage = progress
            feedback_msg.status = f'Moving: {progress:.1f}% complete'
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(0.1)

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.actual_distance = current_distance
            result.message = 'Goal canceled'
            return result

        goal_handle.succeed()
        result.success = True
        result.actual_distance = current_distance
        result.message = 'Navigation completed successfully'
        return result

def main(args=None):
    rclpy.init(args=args)
    navigation_action_server = NavigationActionServer()

    try:
        rclpy.spin(navigation_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 3: Create a Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab2_services_actions',
            executable='robot_state_server',
            name='robot_state_service',
            output='screen'
        ),
        Node(
            package='lab2_services_actions',
            executable='navigation_action_server',
            name='navigation_action_server',
            output='screen'
        )
    ])
```

### Testing Services and Actions

```bash
# List available services
ros2 service list

# Call the robot state service
ros2 service call /get_robot_state lab2_services_actions/srv/RobotState

# List available actions
ros2 action list

# Send a goal to the navigation action
ros2 action send_goal /move_to_goal lab2_services_actions/action/MoveToGoal "{target_distance: 3.0, max_speed: 0.5}"
```

## Expected Outcomes

- IMU data publisher successfully publishing sensor_msgs/Imu messages
- Motor command subscriber successfully receiving geometry_msgs/Twist messages
- Robot state service successfully responding to queries
- Navigation action completing goals with proper feedback
- Understanding of ROS 2 communication patterns

## Acceptance Criteria Met

- [X] Complete launch file examples
- [X] Parameter management techniques
- [X] Lab 1: Basic communication implementation
- [X] Lab 2: Service & Action implementation
