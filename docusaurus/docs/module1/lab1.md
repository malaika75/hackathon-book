# Lab 1: ROS 2 Basic Communication

## Objective

In this lab, you will implement publisher/subscriber communication for simple sensor data (e.g., IMU readings) and motor commands using ROS 2.

## Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Basic understanding of ROS 2 concepts (nodes, topics, messages)
- Python or C++ development environment

## Lab Setup

### Hardware Requirements (Simulation)

If you don't have physical hardware, you can use the following simulation setup:

1. Install the `ros-humble-rosbridge-suite` package for web-based simulation
2. Install `turtlesim` for basic simulation: `sudo apt install ros-humble-turtlesim`

### Creating the Lab Package

```bash
mkdir -p ~/ros2_labs/src
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab1_communication --dependencies rclpy std_msgs geometry_msgs sensor_msgs
cd ~/ros2_labs
colcon build --packages-select lab1_communication
source install/setup.bash
```

## Lab Implementation

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

        # Set timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        # Simulate IMU data with some realistic values
        # In a real robot, these would come from actual IMU sensor
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = math.sin(self.i * 0.1) * 0.1
        msg.orientation.w = math.cos(self.i * 0.1) * 0.1

        # Angular velocity (simulated)
        msg.angular_velocity.x = random.uniform(-0.1, 0.1)
        msg.angular_velocity.y = random.uniform(-0.1, 0.1)
        msg.angular_velocity.z = random.uniform(-0.2, 0.2)

        # Linear acceleration (simulated)
        msg.linear_acceleration.x = random.uniform(-1.0, 1.0)
        msg.linear_acceleration.y = random.uniform(-1.0, 1.0)
        msg.linear_acceleration.z = 9.81 + random.uniform(-0.5, 0.5)  # Gravity + noise

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
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Motor subscriber node initialized')

    def listener_callback(self, msg):
        # In a real robot, this would control the actual motors
        # For simulation, we'll just log the commands
        self.get_logger().info(f'Received motor command - Linear: ({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}), '
                              f'Angular: ({msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f})')

        # Simulate motor response time
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
        # IMU Publisher Node
        Node(
            package='lab1_communication',
            executable='imu_publisher',
            name='imu_publisher',
            output='screen'
        ),

        # Motor Subscriber Node
        Node(
            package='lab1_communication',
            executable='motor_subscriber',
            name='motor_subscriber',
            output='screen'
        ),

        # Optional: Add a teleoperation node to send motor commands
        ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '/cmd_vel', 'geometry_msgs/msg/Twist',
                 '{linear: {x: 0.5}, angular: {z: 0.2}}', '-r', '1'],
            output='screen'
        )
    ])
```

## Lab Execution

### Building and Running

1. Build the package:
   ```bash
   cd ~/ros2_labs
   colcon build --packages-select lab1_communication
   source install/setup.bash
   ```

2. Run the launch file:
   ```bash
   ros2 launch lab1_communication lab1_launch.py
   ```

### Testing Communication

Open a new terminal and test communication:

```bash
# Check active topics
ros2 topic list

# Echo IMU data
ros2 topic echo /imu_data

# Send motor commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'
```

## Expected Outcomes

- IMU data publisher successfully publishing sensor_msgs/Imu messages
- Motor command subscriber successfully receiving geometry_msgs/Twist messages
- Proper use of ROS 2 communication patterns
- Understanding of message types and topic-based communication

## Troubleshooting

1. **Node not found**: Make sure to source the workspace: `source install/setup.bash`
2. **Permission errors**: Ensure Python files have execute permissions: `chmod +x *.py`
3. **Topic not connecting**: Check that both nodes are on the same ROS_DOMAIN_ID

## Solution Guide for Instructors

### Key Learning Points
- Understanding publisher-subscriber pattern
- Working with standard message types
- Using launch files for system orchestration
- Basic ROS 2 node implementation

### Assessment Criteria
- Correct implementation of publisher and subscriber nodes
- Proper message type usage
- Successful communication between nodes
- Understanding of ROS 2 concepts demonstrated

## Acceptance Criteria Met

- [X] Complete Lab 1 instructions with expected outcomes
- [X] Solution guides for instructors