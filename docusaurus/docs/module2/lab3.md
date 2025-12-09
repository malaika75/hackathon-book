# Lab 3: Robot Model in Simulation

## Objective

In this lab, you will create a URDF model of a simple robot and spawn it in Gazebo simulation. This lab focuses on understanding robot description formats and their integration with simulation environments.

## Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Gazebo installation
- Basic understanding of URDF and ROS 2 concepts

## Lab Setup

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab3_robot_simulation --dependencies rclpy std_msgs geometry_msgs sensor_msgs xacro
cd ~/ros2_labs
colcon build --packages-select lab3_robot_simulation
source install/setup.bash
```

### Creating Directories

```bash
cd ~/ros2_labs/src/lab3_robot_simulation
mkdir -p urdf meshes config launch worlds
```

## Lab Implementation

### Task 1: Create a Simple Mobile Robot URDF

Create the main robot URDF file `urdf/simple_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_robot">
  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="0.15" />

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia
        ixx="1.0" ixy="0.0" ixz="0.0"
        iyy="1.0" iyz="0.0"
        izz="1.0"/>
    </inertial>
  </link>

  <!-- Define wheels using macro -->
  <xacro:macro name="wheel" params="prefix reflect_x reflect_y">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1"/>
        <inertia
          ixx="0.1" ixy="0.0" ixz="0.0"
          iyy="0.1" iyz="0.0"
          izz="0.1"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${reflect_x * base_length/2} ${reflect_y * (base_width/2 + wheel_width/2)} ${-wheel_radius}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Create wheels -->
  <xacro:wheel prefix="front_left" reflect_x="1" reflect_y="1"/>
  <xacro:wheel prefix="front_right" reflect_x="1" reflect_y="-1"/>
  <xacro:wheel prefix="back_left" reflect_x="-1" reflect_y="1"/>
  <xacro:wheel prefix="back_right" reflect_x="-1" reflect_y="-1"/>

  <!-- Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="${base_length/2 - 0.025} 0 ${base_height/2}" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <update_rate>30</update_rate>
      <left_joint>front_left_wheel_joint</left_joint>
      <right_joint>front_right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>

  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/simple_robot</namespace>
          <remapping>~/image_raw:=/camera/image_raw</remapping>
          <remapping>~/camera_info:=/camera/camera_info</remapping>
        </ros>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="front_left_wheel">
    <material>Gazebo/Black</material>
  </gazebo>
</robot>
```

### Task 2: Create a Launch File

Create the launch file `launch/simple_robot.launch.py`:

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
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': Command([
                'xacro ',
                PathJoinSubstitution([
                    FindPackageShare('lab3_robot_simulation'),
                    'urdf',
                    'simple_robot.urdf.xacro'
                ])
            ])
        }]
    )

    # Spawn entity in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_robot',
            '-x', '0', '-y', '0', '-z', '0.1'
        ],
        output='screen'
    )

    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'verbose': 'false',
            'pause': 'false'
        }.items()
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_state_publisher,
        gazebo,
        spawn_entity
    ])
```

We need to add the missing import for Command:

```python
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
```

Actually, let me fix the launch file:

```python
# launch/simple_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': Command([
                'xacro ',
                PathJoinSubstitution([
                    FindPackageShare('lab3_robot_simulation'),
                    'urdf',
                    'simple_robot.urdf.xacro'
                ])
            ])
        }]
    )

    # Spawn entity in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_robot',
            '-x', '0', '-y', '0', '-z', '0.1'
        ],
        output='screen'
    )

    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'verbose': 'false',
            'pause': 'false'
        }.items()
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_state_publisher,
        gazebo,
        spawn_entity
    ])
```

### Task 3: Create a Test Controller

Create a simple controller to test the robot in `lab3_robot_simulation/simple_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer to send commands
        self.timer = self.create_timer(0.1, self.send_command)
        self.i = 0

        self.get_logger().info('Simple Controller initialized')

    def send_command(self):
        msg = Twist()

        # Create a square movement pattern
        if self.i < 50:  # Move forward
            msg.linear.x = 0.5
            msg.angular.z = 0.0
        elif self.i < 100:  # Turn right
            msg.linear.x = 0.0
            msg.angular.z = -0.5
        elif self.i < 150:  # Move forward
            msg.linear.x = 0.5
            msg.angular.z = 0.0
        elif self.i < 200:  # Turn right
            msg.linear.x = 0.0
            msg.angular.z = -0.5
        else:
            self.i = 0  # Reset counter
            return

        self.publisher_.publish(msg)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    simple_controller = SimpleController()

    try:
        rclpy.spin(simple_controller)
    except KeyboardInterrupt:
        pass
    finally:
        simple_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab Execution

### Building and Running

1. Build the package:
   ```bash
   cd ~/ros2_labs
   colcon build --packages-select lab3_robot_simulation
   source install/setup.bash
   ```

2. Run the launch file:
   ```bash
   ros2 launch lab3_robot_simulation simple_robot.launch.py
   ```

3. In a new terminal, run the controller:
   ```bash
   ros2 run lab3_robot_simulation simple_controller
   ```

### Verification Steps

1. **Check TF Tree**:
   ```bash
   ros2 run tf2_tools view_frames
   ```

2. **View Robot Model**:
   ```bash
   ros2 run rviz2 rviz2
   # Add RobotModel display and set Fixed Frame to 'base_link'
   ```

3. **Check Topics**:
   ```bash
   ros2 topic list
   ros2 topic echo /odom
   ros2 topic echo /camera/image_raw --field data
   ```

## Expected Outcomes

- Robot model successfully loaded in Gazebo
- Robot responds to velocity commands
- TF tree shows proper kinematic relationships
- Camera sensor publishing image data
- Proper URDF structure with visual, collision, and inertial properties

## Troubleshooting

1. **Robot not spawning**: Check that Gazebo is running and xacro file is valid
2. **No movement**: Verify joint names match between URDF and controller
3. **TF issues**: Ensure proper frame relationships in URDF
4. **Xacro errors**: Check syntax and property definitions

## Solution Guide for Instructors

### Key Learning Points
- Understanding URDF structure and components
- Creating parametric robot models with xacro
- Integrating sensors and actuators in simulation
- Using launch files for complex system startup

### Assessment Criteria
- Correct URDF structure with all required elements
- Proper integration with Gazebo simulation
- Successful robot spawning and control
- Understanding of coordinate frames and transformations

## Acceptance Criteria Met

- [X] Complete Lab 3 instructions with expected outcomes
- [X] Solution guides for instructors