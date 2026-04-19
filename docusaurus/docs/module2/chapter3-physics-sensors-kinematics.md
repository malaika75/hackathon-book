# Chapter 3: Physics, Sensors & Kinematics

This chapter covers physics engines, sensor simulation, and kinematics in robotic simulation.

## Physics and Sensor Simulation

### Overview

Physics simulation and sensor modeling are critical components of realistic robotic simulation. This section covers how to configure physics engines and simulate various sensors in Gazebo.

## Physics Engines

### Available Physics Engines

Gazebo supports multiple physics engines:

#### ODE (Open Dynamics Engine)
- **Pros**: Fast, stable for most applications
- **Cons**: Can be less accurate for complex contacts
- **Best for**: General-purpose simulation, mobile robots

#### Bullet
- **Pros**: Good balance of speed and accuracy
- **Cons**: Can be less stable with complex constraints
- **Best for**: Manipulation tasks, complex contacts

#### Simbody
- **Pros**: Very accurate for complex multi-body systems
- **Cons**: Slower performance
- **Best for**: Humanoid robots, complex mechanisms

### Physics Engine Configuration

In your world file, specify the physics engine:

```xml
<sdf version="1.7">
  <world name="default">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

### Physics Parameters

- **max_step_size**: Simulation time step (smaller = more accurate but slower)
- **real_time_factor**: Target simulation speed relative to real time
- **real_time_update_rate**: Updates per second

## Sensor Simulation

### Types of Sensors

#### 1. Camera Sensors

Camera simulation in URDF:

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/image_raw:=/camera/image_raw</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

#### 2. LiDAR/Depth Sensors

LiDAR simulation:

```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=/scan</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

#### 3. IMU Sensors

IMU simulation:

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
      </angular_velocity>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=/imu</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

### Sensor Noise and Accuracy

Add realistic noise to sensors:

```xml
<ray>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>
  </noise>
</ray>
```

---

## Kinematics in Simulation

### Overview

Kinematics is the study of motion without considering the forces that cause it. In robotics simulation, kinematics is essential for understanding how robot joints and links move in relation to each other.

## Forward Kinematics in Simulation

### Understanding Forward Kinematics

Forward kinematics (FK) is the process of calculating the position and orientation of a robot's end-effector based on the joint angles.

### Mathematical Foundation

For a robotic arm with n joints:
```
T = A1(θ1) × A2(θ2) × ... × An(θn)
```

Where T is the final transformation matrix and Ai(θi) represents the transformation due to joint i.

### Forward Kinematics in URDF

In URDF, the kinematic chain is defined through joints and links:

```xml
<robot name="kinematic_robot">
  <link name="base_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <link name="link1"/>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="link2"/>
</robot>
```

## Inverse Kinematics in Simulation

### Understanding Inverse Kinematics

Inverse kinematics (IK) is the reverse process - computing the joint angles required to achieve a desired end-effector position and orientation.

### IK Solvers

1. **Analytical Solutions**: Closed-form solutions for simple kinematic chains
2. **Numerical Methods**: Iterative approaches for complex robots
3. **Jacobian-based Methods**: Using the Jacobian matrix

### Using MoveIt for IK in Simulation

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped

class IKSolver(Node):
    def __init__(self):
        super().__init__('ik_solver')
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available...')

    def solve_ik(self, x, y, z):
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'manipulator'

        target_pose = PoseStamped()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        target_pose.pose.orientation.w = 1.0

        request.ik_request.pose_stamped = target_pose
        request.ik_request.timeout.sec = 1

        future = self.ik_client.call_async(request)
        return future

def main(args=None):
    rclpy.init(args=args)
    ik_solver = IKSolver()
    rclpy.spin(ik_solver)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Kinematics Problems

### Problem: Reaching a Target Position

```python
#!/usr/bin/env python3

import math

class KinematicController:
    def __init__(self):
        self.l1 = 0.5  # Length of first link
        self.l2 = 0.4  # Length of second link

    def inverse_kinematics(self, x, y):
        """Solve inverse kinematics for 2-DOF planar arm"""
        r = math.sqrt(x*x + y*y)

        if r > (self.l1 + self.l2):
            return None  # Not reachable
        if r < abs(self.l1 - self.l2):
            return None  # Inside workspace

        cos_theta2 = (self.l1**2 + self.l2**2 - r**2) / (2 * self.l1 * self.l2)
        theta2 = math.acos(max(-1, min(1, cos_theta2)))

        k1 = self.l1 + self.l2 * math.cos(theta2)
        k2 = self.l2 * math.sin(theta2)
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        return [theta1, theta2]

    def forward_kinematics(self, theta1, theta2):
        """Calculate forward kinematics"""
        x = self.l1 * math.cos(theta1) + self.l2 * math.cos(theta1 + theta2)
        y = self.l1 * math.sin(theta1) + self.l2 * math.sin(theta1 + theta2)
        return [x, y]
```

### Trajectory Following

```python
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def send_trajectory(joint_names, positions):
    traj_msg = JointTrajectory()
    traj_msg.joint_names = joint_names

    for i, pos in enumerate(positions):
        traj_point = JointTrajectoryPoint()
        traj_point.positions = pos
        traj_point.time_from_start.sec = i * 2
        traj_msg.points.append(traj_point)

    return traj_msg
```

## Best Practices

### For Physics Simulation
- Start with default parameters and adjust as needed
- Use realistic mass and inertia values
- Test with different physics engines
- Monitor simulation real-time factor

### For Sensor Simulation
- Add realistic noise models to match real sensors
- Validate sensor data against real hardware
- Use appropriate update rates

### For Kinematics
- Ensure URDF/SDF models match real robot kinematics
- Consider multiple IK solutions
- Check for kinematic singularities

## Acceptance Criteria Met

- [X] Physics engine parameter explanations
- [X] Sensor simulation examples
- [X] Forward and inverse kinematics
- [X] Practical kinematics problems
