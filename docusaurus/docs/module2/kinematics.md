# Kinematics in Simulation

## Overview

Kinematics is the study of motion without considering the forces that cause it. In robotics simulation, kinematics is essential for understanding how robot joints and links move in relation to each other. This includes both forward kinematics (computing end-effector position from joint angles) and inverse kinematics (computing joint angles from desired end-effector position).

## Forward Kinematics in Simulation

### Understanding Forward Kinematics

Forward kinematics (FK) is the process of calculating the position and orientation of a robot's end-effector based on the joint angles. In simulation, this is used to visualize and validate robot movements.

### Mathematical Foundation

For a robotic arm with n joints, the forward kinematics can be computed using the Denavit-Hartenberg (DH) parameters or transformation matrices:

```
T = A1(θ1) × A2(θ2) × ... × An(θn)
```

Where T is the final transformation matrix and Ai(θi) represents the transformation due to joint i.

### Forward Kinematics in URDF

In URDF, the kinematic chain is defined through joints and links. The robot's pose is calculated based on these definitions:

```xml
<robot name="kinematic_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- First joint and link -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <link name="link1">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
  </link>

  <!-- Second joint and link -->
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="link2">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- End effector -->
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="end_effector"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <link name="end_effector"/>
</robot>
```

## Inverse Kinematics in Simulation

### Understanding Inverse Kinematics

Inverse kinematics (IK) is the reverse process of forward kinematics - computing the joint angles required to achieve a desired end-effector position and orientation. This is crucial for task-based robot control.

### IK Solvers

Several approaches exist for solving inverse kinematics:

1. **Analytical Solutions**: Closed-form solutions for simple kinematic chains
2. **Numerical Methods**: Iterative approaches for complex robots
3. **Jacobian-based Methods**: Using the Jacobian matrix to relate joint velocities to end-effector velocities

### Using MoveIt for IK in Simulation

MoveIt is the standard motion planning framework for ROS that includes sophisticated IK solvers:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class IKSolver(Node):
    def __init__(self):
        super().__init__('ik_solver')

        # Create client for inverse kinematics service
        self.ik_client = self.create_client(
            GetPositionIK, 'compute_ik')

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting again...')

        # Timer to periodically solve IK
        self.timer = self.create_timer(1.0, self.solve_ik)

        self.get_logger().info('IK Solver initialized')

    def solve_ik(self):
        # Create IK request
        request = GetPositionIK.Request()

        # Set the group name (defined in SRDF)
        request.ik_request.group_name = 'manipulator'

        # Set the target pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose.position.x = 0.5
        target_pose.pose.position.y = 0.0
        target_pose.pose.position.z = 0.5
        target_pose.pose.orientation.w = 1.0  # No rotation

        request.ik_request.pose_stamped = target_pose
        request.ik_request.timeout.sec = 1  # 1 second timeout

        # Call the IK service
        future = self.ik_client.call_async(request)
        future.add_done_callback(self.ik_callback)

    def ik_callback(self, future):
        try:
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                self.get_logger().info(
                    f'IK Solution found: {len(response.solution.joint_state.name)} joints')
                for i, name in enumerate(response.solution.joint_state.name):
                    self.get_logger().info(
                        f'{name}: {response.solution.joint_state.position[i]:.3f}')
            else:
                self.get_logger().info(f'IK failed with error code: {response.error_code.val}')
        except Exception as e:
            self.get_logger().error(f'Exception in IK callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    ik_solver = IKSolver()

    try:
        rclpy.spin(ik_solver)
    except KeyboardInterrupt:
        pass
    finally:
        ik_solver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation-Based Kinematics Tools

### RViz Visualization

RViz can be used to visualize kinematic chains and transformations:

```bash
# Launch RViz with robot model
ros2 run rviz2 rviz2
# Add RobotModel display and set your robot description parameter
```

### TF Tree Analysis

Use ROS 2 tools to analyze the kinematic tree:

```bash
# View the TF tree
ros2 run tf2_tools view_frames

# Echo transforms
ros2 run tf2_ros tf2_echo base_link end_effector

# View TF in RViz
ros2 run rviz2 rviz2
# Add TF display
```

## Practical Kinematics Problems Solved

### Problem 1: Reaching a Target Position

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np
import math

class KinematicController(Node):
    def __init__(self):
        super().__init__('kinematic_controller')

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot parameters (simple 2-DOF arm)
        self.l1 = 0.5  # Length of first link
        self.l2 = 0.4  # Length of second link

        # Current joint angles
        self.joint_angles = [0.0, 0.0]

        # Target position
        self.target_x = 0.6
        self.target_y = 0.4

        self.get_logger().info('Kinematic Controller initialized')

    def inverse_kinematics(self, x, y):
        """Solve inverse kinematics for 2-DOF planar arm"""
        # Calculate distance to target
        r = math.sqrt(x*x + y*y)

        # Check if target is reachable
        if r > (self.l1 + self.l2):
            self.get_logger().warn('Target position is not reachable')
            return None

        if r < abs(self.l1 - self.l2):
            self.get_logger().warn('Target position is inside workspace')
            return None

        # Calculate second joint angle
        cos_theta2 = (self.l1**2 + self.l2**2 - r**2) / (2 * self.l1 * self.l2)
        theta2 = math.acos(max(-1, min(1, cos_theta2)))  # Clamp to [-1, 1]

        # Calculate first joint angle
        k1 = self.l1 + self.l2 * math.cos(theta2)
        k2 = self.l2 * math.sin(theta2)

        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        return [theta1, theta2]

    def forward_kinematics(self, theta1, theta2):
        """Calculate forward kinematics for 2-DOF planar arm"""
        x = self.l1 * math.cos(theta1) + self.l2 * math.cos(theta1 + theta2)
        y = self.l1 * math.sin(theta1) + self.l2 * math.sin(theta1 + theta2)
        return [x, y]

    def control_loop(self):
        # Solve inverse kinematics for target position
        solution = self.inverse_kinematics(self.target_x, self.target_y)

        if solution:
            # Create joint state message
            joint_msg = JointState()
            joint_msg.name = ['joint1', 'joint2']
            joint_msg.position = solution
            joint_msg.velocity = [0.0, 0.0]
            joint_msg.effort = [0.0, 0.0]

            # Publish joint commands
            self.joint_pub.publish(joint_msg)

            # Calculate actual position using forward kinematics
            actual_pos = self.forward_kinematics(solution[0], solution[1])
            self.get_logger().info(
                f'Target: ({self.target_x:.2f}, {self.target_y:.2f}), '
                f'Actual: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f})')

def main(args=None):
    rclpy.init(args=args)
    controller = KinematicController()

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

### Problem 2: Trajectory Following

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import math

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')

        # Publisher for trajectory commands
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10)

        # Timer to send trajectory
        self.timer = self.create_timer(2.0, self.send_trajectory)

        self.get_logger().info('Trajectory Controller initialized')

    def send_trajectory(self):
        """Send a trajectory to move through several points"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['joint1', 'joint2']

        # Define trajectory points
        points = [
            [0.0, 0.0],      # Start position
            [0.5, 0.3],      # Midpoint 1
            [0.0, 0.7],      # Midpoint 2
            [0.0, 0.0]       # Return to start
        ]

        time_from_start = 0.0
        for i, point in enumerate(points):
            traj_point = JointTrajectoryPoint()
            traj_point.positions = point
            time_from_start += 2.0  # 2 seconds per point
            traj_point.time_from_start.sec = int(time_from_start)
            traj_point.time_from_start.nanosec = int(
                (time_from_start - int(time_from_start)) * 1e9)

            traj_msg.points.append(traj_point)

        # Publish trajectory
        self.traj_pub.publish(traj_msg)
        self.get_logger().info(f'Published trajectory with {len(points)} points')

def main(args=None):
    rclpy.init(args=args)
    controller = TrajectoryController()

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

## Kinematics in Different Simulation Platforms

### Gazebo Kinematics

Gazebo handles kinematics through its physics engine. For accurate kinematic simulation:

1. **Proper Joint Limits**: Define realistic joint limits in URDF/SDF
2. **Inertial Properties**: Accurate mass and inertia for realistic movement
3. **Transmission Plugins**: Use appropriate transmission plugins for joint control

### Webots Kinematics

Webots provides built-in kinematic solvers:

```python
# Webots example (Python controller)
from controller import Robot, Motor, PositionSensor

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get motor and position sensor handles
motor1 = robot.getDevice('motor1')
motor2 = robot.getDevice('motor2')

pos_sensor1 = robot.getDevice('pos_sensor1')
pos_sensor1.enable(timestep)

# Set motor positions (inverse kinematics result)
motor1.setPosition(target_angle1)
motor2.setPosition(target_angle2)
```

## Best Practices for Kinematic Simulation

### For Forward Kinematics
1. **Accurate Models**: Ensure URDF/SDF models match real robot kinematics
2. **Consistent Frames**: Use consistent coordinate frame definitions
3. **Validation**: Compare simulation results with analytical solutions

### For Inverse Kinematics
1. **Multiple Solutions**: Consider that IK may have multiple valid solutions
2. **Singularity Handling**: Implement checks for kinematic singularities
3. **Smooth Transitions**: Ensure smooth joint movements between targets
4. **Workspace Limits**: Check if targets are within the robot's workspace

### Performance Considerations
- Use efficient IK solvers for real-time applications
- Consider pre-computing kinematic solutions when possible
- Validate solutions to avoid unreachable configurations

## Acceptance Criteria Met

- [X] FK and IK concepts explained with examples
- [X] Simulation-based kinematics tools
- [X] Practical kinematics problems solved