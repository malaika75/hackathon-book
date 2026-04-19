# Chapter 2: Robot Description & ROS Simulation Integration

This chapter covers robot description formats (URDF/SDF) and ROS 2 simulation integration.

## Robot Description Formats: URDF and SDF

### Introduction

Robot description formats are essential for defining robot models in simulation and real-world applications. The two primary formats in robotics are URDF (Unified Robot Description Format) for ROS-based systems and SDF (Simulation Description Format) for Gazebo and other simulators.

## URDF (Unified Robot Description Format)

URDF is the standard format for representing robot models in ROS. It's an XML-based format that describes the physical and kinematic properties of a robot.

### URDF Structure

A typical URDF file includes:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define the physical parts of the robot -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links together -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.2 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

### Key URDF Elements

#### Links
- **visual**: Defines how the link looks in visualization
- **collision**: Defines collision properties for physics simulation
- **inertial**: Defines mass and inertia properties for physics simulation

#### Joints
- **fixed**: No degrees of freedom (0 DOF)
- **revolute**: Rotational joint with limits (1 DOF)
- **continuous**: Rotational joint without limits (1 DOF)
- **prismatic**: Linear sliding joint with limits (1 DOF)
- **floating**: 6 DOF (x, y, z, roll, pitch, yaw)
- **planar**: Movement on a plane (2 DOF)

### URDF Best Practices

1. **Use Proper Inertial Properties**: Accurate mass and inertia values are crucial for realistic simulation
2. **Separate Visual and Collision Geometry**: Use simple shapes for collision detection, complex meshes for visualization
3. **Consistent Naming**: Use descriptive names for links and joints
4. **Validate URDF**: Use tools like `check_urdf` to validate your URDF files

## SDF (Simulation Description Format)

SDF is the native format for Gazebo and other simulators. It provides more features than URDF, including support for multiple robots in one file and more complex sensor models.

### SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <pose>0 0 0.5 0 0 0</pose>

    <link name="chassis">
      <pose>0 0 0 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.2</iyy><iyz>0</iyz>
          <izz>0.3</izz>
        </inertia>
      </inertial>
    </link>

    <joint name="chassis_wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>
  </model>
</sdf>
```

### SDF Advantages Over URDF

- Support for multiple models in one file
- More advanced sensor simulation
- Better support for complex environments
- More physics engine options
- Native Gazebo integration

## URDF vs SDF Comparison

| Feature | URDF | SDF |
|---------|------|-----|
| Primary Use | ROS ecosystem | Gazebo/simulation |
| Complexity | Simpler, ROS-focused | More comprehensive |
| Multi-robot Support | Requires extensions | Native support |
| Sensor Modeling | Limited | Advanced |
| Physics Engines | Limited | Multiple options |
| ROS Integration | Excellent | Requires plugins |

## Model Validation Techniques

### URDF Validation

1. **Syntax Check**:
   ```bash
   check_urdf /path/to/robot.urdf
   ```

2. **Visualization**:
   ```bash
   ros2 run rviz2 rviz2
   # Add RobotModel display and load your URDF
   ```

### SDF Validation

1. **Gazebo Integration**:
   ```bash
   gazebo -s libgazebo_ros_factory.so
   ```

## Using xacro for Complex URDFs

For complex robots, use xacro (XML Macros) to simplify URDF creation:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <xacro:macro name="wheel" params="prefix parent xyz">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="0 ${M_PI/2} 0"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
      </visual>
    </link>
  </xacro:macro>

  <link name="base_link"/>
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.2 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.2 0"/>
</robot>
```

---

## ROS 2 Simulation Integration

### Overview

Integrating ROS 2 with simulation environments is crucial for developing and testing robotic applications. This integration allows you to control simulated robots using ROS 2 nodes, topics, and services, just like you would with real hardware.

### Gazebo-ROS Bridge

The Gazebo-ROS bridge provides the connection between Gazebo simulation and ROS 2.

### Installation

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins
```

### Key Components

1. **gazebo_ros_factory**: Spawns and deletes models in simulation
2. **gazebo_ros_init**: Initializes the ROS interface for Gazebo
3. **gazebo_ros_paths**: Handles model path resolution

### Creating a Robot Model with ROS 2 Plugins

To make your robot controllable from ROS 2, add Gazebo plugins to your URDF:

```xml
<robot name="my_robot">
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
    </plugin>
  </gazebo>
</robot>
```

### Controller Configuration

Create a controller configuration file:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100
    use_sim_time: true

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController
```

### Example Robot Control in Simulation

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

        self.cmd_vel_pub = self.create_publisher(Twist, '/my_robot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/my_robot/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.current_pose = [0.0, 0.0, 0.0]
        self.target_pose = [1.0, 1.0, 0.0]

    def odom_callback(self, msg):
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

    def control_loop(self):
        cmd_msg = Twist()
        dx = self.target_pose[0] - self.current_pose[0]
        dy = self.target_pose[1] - self.current_pose[1]
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0.1:
            cmd_msg.linear.x = min(0.5, distance * 0.5)
        else:
            cmd_msg.linear.x = 0.0

        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = SimulationController()
    rclpy.spin(controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Common Integration Issues

1. **Time Synchronization**: Set `use_sim_time:=true` when running in simulation
2. **TF Tree Issues**: Ensure proper frame relationships in URDF
3. **Controller Issues**: Verify joint names match between URDF and configuration

### Using ros2 control Commands

```bash
# List controllers
ros2 control list_controllers

# Switch controllers
ros2 control switch_controllers --activate joint_state_broadcaster
```

## Best Practices

1. **Use Simulation Time**: Always set `use_sim_time:=true`
2. **Proper TF Frames**: Ensure complete TF tree
3. **Controller Validation**: Test controllers in simulation before hardware
4. **Parameter Tuning**: Adjust PID values specifically for simulation

## Acceptance Criteria Met

- [X] Complete URDF examples with explanations
- [X] SDF comparison and use cases
- [X] ROS 2 simulation integration
- [X] Controller configuration examples
