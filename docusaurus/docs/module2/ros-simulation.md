# ROS 2 Simulation Integration

## Overview

Integrating ROS 2 with simulation environments is crucial for developing and testing robotic applications. This integration allows you to control simulated robots using ROS 2 nodes, topics, and services, just like you would with real hardware.

## Gazebo-ROS Bridge

The Gazebo-ROS bridge provides the connection between Gazebo simulation and ROS 2. It allows ROS 2 nodes to interact with simulated robots and environments.

### Installation

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins
```

### Key Components

1. **gazebo_ros_factory**: Spawns and deletes models in simulation
2. **gazebo_ros_init**: Initializes the ROS interface for Gazebo
3. **gazebo_ros_paths**: Handles model path resolution

## Step-by-Step Integration Guide

### 1. Setting Up the Simulation Environment

First, create a launch file to start Gazebo with ROS 2 integration:

```python
# launch/robot_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    world = LaunchConfiguration('world')
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/gazebo_worlds`'
    )

    # Start Gazebo with ROS 2 plugins
    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so', world],
        output='screen'
    )

    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    return LaunchDescription([
        world_arg,
        gzserver_cmd,
        gzclient_cmd
    ])
```

### 2. Creating a Robot Model with ROS 2 Plugins

To make your robot controllable from ROS 2, you need to add Gazebo plugins to your URDF:

```xml
<!-- In your robot URDF file -->
<robot name="my_robot">
  <!-- Robot links and joints as defined previously -->

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Differential drive plugin for mobile robots -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <update_rate>30</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.1</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

  <!-- Joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <ros>
        <namespace>/my_robot</namespace>
      </ros>
      <update_rate>30</update_rate>
      <joint_name>left_wheel_joint</joint_name>
      <joint_name>right_wheel_joint</joint_name>
    </plugin>
  </gazebo>
</robot>
```

### 3. Controller Configuration

Create a controller configuration file (`config/my_robot_controllers.yaml`):

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100
    use_sim_time: true

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    position_controller:
      type: position_controllers/JointGroupPositionController

velocity_controller:
  ros__parameters:
    joints:
      - left_wheel_joint
      - right_wheel_joint

position_controller:
  ros__parameters:
    joints:
      - left_wheel_joint
      - right_wheel_joint
```

## Example Robot Control in Simulation

### Python Node for Robot Control

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class SimulationController(Node):
    def __init__(self):
        super().__init__('simulation_controller')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/my_robot/cmd_vel', 10)

        # Subscriber for odometry
        self.odom_sub = self.create_subscription(
            Odometry, '/my_robot/odom', self.odom_callback, 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.current_pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.target_pose = [1.0, 1.0, 0.0]   # x, y, theta
        self.get_logger().info('Simulation controller initialized')

    def odom_callback(self, msg):
        # Update current pose from odometry
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

        # Convert quaternion to euler for theta
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_pose[2] = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        # Simple control to move toward target
        cmd_msg = Twist()

        dx = self.target_pose[0] - self.current_pose[0]
        dy = self.target_pose[1] - self.current_pose[1]
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0.1:  # If not close to target
            target_angle = math.atan2(dy, dx)
            angle_diff = target_angle - self.current_pose[2]

            # Normalize angle
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Simple proportional controller
            cmd_msg.linear.x = min(0.5, distance * 0.5)
            cmd_msg.angular.z = angle_diff * 1.0
        else:
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = SimulationController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Common Integration Issues and Solutions

### 1. Time Synchronization

In simulation, it's important to use simulation time:

```python
# In your launch file
use_sim_time_param = DeclareLaunchArgument(
    'use_sim_time',
    default_value='true',
    description='Use simulation (Gazebo) clock if true'
)

# In your node
use_sim_time = LaunchConfiguration('use_sim_time')
# Pass this to your nodes as a parameter
```

### 2. TF Tree Issues

Make sure your robot has proper TF frames:

```xml
<!-- In your URDF -->
<link name="odom"/>
<link name="base_link"/>

<joint name="odom_base_joint" type="fixed">
  <parent link="odom"/>
  <child link="base_link"/>
</joint>
```

### 3. Controller Issues

If controllers aren't loading, check:

1. Controller configuration file syntax
2. Controller manager is running
3. Joint names match between URDF and configuration
4. Use `ros2 control` commands to check status:

```bash
# List controllers
ros2 control list_controllers

# Switch controllers
ros2 control switch_controllers --activate joint_state_broadcaster
```

## Best Practices

### For Simulation Integration

1. **Use Simulation Time**: Always set `use_sim_time:=true` when running in simulation
2. **Proper TF Frames**: Ensure your robot has a complete TF tree
3. **Controller Validation**: Test controllers in simulation before hardware
4. **Parameter Tuning**: Adjust PID values specifically for simulation
5. **Sensor Accuracy**: Account for sensor noise and limitations in simulation

### Performance Considerations

- Use appropriate physics engine parameters for your application
- Limit the number of active controllers during development
- Use simplified collision models for better performance
- Consider using `--headless` option for faster simulation without GUI

## Acceptance Criteria Met

- [X] Step-by-step integration guide
- [X] Example robot control in simulation
- [X] Common integration issues and solutions