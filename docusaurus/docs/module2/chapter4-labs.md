# Chapter 4: Labs - Robot Models & Sensor Integration

This chapter contains hands-on labs for creating robot models in simulation and integrating sensors.

---

## Lab 3: Robot Model in Simulation

### Objective

In this lab, you will create a URDF model of a simple robot and spawn it in Gazebo simulation.

### Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Gazebo installation
- Basic understanding of URDF and ROS 2 concepts

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab3_robot_simulation --dependencies rclpy std_msgs geometry_msgs sensor_msgs xacro
cd ~/ros2_labs
colcon build --packages-select lab3_robot_simulation
source install/setup.bash
```

### Task 1: Create a Simple Mobile Robot URDF

Create the main robot URDF file `urdf/simple_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_robot">
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
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Wheel macro -->
  <xacro:macro name="wheel" params="prefix reflect_x reflect_y">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
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
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>
</robot>
```

### Task 2: Create a Launch File

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': Command([
                'xacro ',
                PathJoinSubstitution([
                    FindPackageShare('lab3_robot_simulation'),
                    'urdf', 'simple_robot.urdf.xacro'
                ])
            ])
        }]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
        ])
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'simple_robot', '-x', '0', '-y', '0', '-z', '0.1'],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        robot_state_publisher, gazebo, spawn_entity
    ])
```

### Task 3: Create a Test Controller

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.send_command)
        self.i = 0

    def send_command(self):
        msg = Twist()
        if self.i < 50:
            msg.linear.x = 0.5
        elif self.i < 100:
            msg.angular.z = -0.5
        else:
            self.i = 0
            return
        self.publisher_.publish(msg)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    simple_controller = SimpleController()
    rclpy.spin(simple_controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Execution

```bash
colcon build --packages-select lab3_robot_simulation
source install/setup.bash
ros2 launch lab3_robot_simulation simple_robot.launch.py
ros2 run lab3_robot_simulation simple_controller
```

### Expected Outcomes

- Robot model successfully loaded in Gazebo
- Robot responds to velocity commands
- TF tree shows proper kinematic relationships

---

## Lab 4: Sensor Integration & Data Acquisition

### Objective

In this lab, you will simulate lidar, camera, and IMU sensors, visualize data in RViz, and record data using ros2 bag.

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab4_sensor_integration --dependencies rclpy sensor_msgs geometry_msgs
cd ~/ros2_labs
colcon build
source install/setup.bash
```

### Task 1: Create Robot with Multiple Sensors

Create an enhanced robot URDF with sensors in `urdf/sensor_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="sensor_robot">
  <!-- Base, wheels (same as Lab 3) -->

  <!-- Lidar sensor -->
  <link name="lidar_link">
    <visual>
      <cylinder radius="0.05" length="0.05"/>
    </visual>
  </link>
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Lidar plugin -->
  <gazebo reference="lidar_link">
    <sensor name="lidar" type="ray">
      <ray>
        <scan><horizontal><samples>360</samples></horizontal></scan>
        <range><min>0.1</min><max>10.0</max></range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <ros><remapping>~/out:=/scan</remapping></ros>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Camera -->
  <link name="camera_link">
    <visual><box size="0.05 0.05 0.05"/></visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0.1 0.1" rpy="0 0 0"/>
  </joint>
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <camera><image><width>640</width><height>480</height></image></camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros><remapping>~/image_raw:=/camera/image_raw</remapping></ros>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU -->
  <link name="imu_link">
    <visual><box size="0.02 0.02 0.02"/></visual>
  </link>
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <update_rate>100</update_rate>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
        <ros><remapping>~/out:=/imu</remapping></ros>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Task 2: Robot Controller with Obstacle Avoidance

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.obstacle_distance = float('inf')

    def scan_callback(self, msg):
        front_distances = []
        for i in range(len(msg.ranges)):
            angle = msg.angle_min + i * msg.angle_increment
            if -math.pi/4 <= angle <= math.pi/4:
                if not math.isnan(msg.ranges[i]) and msg.ranges[i] > 0:
                    front_distances.append(msg.ranges[i])
        if front_distances:
            self.obstacle_distance = min(front_distances)

    def control_loop(self):
        cmd_msg = Twist()
        if self.obstacle_distance < 0.5:
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5
        else:
            cmd_msg.linear.x = 0.3
        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    rclpy.spin(controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 3: Data Recording

```bash
# Record sensor data
ros2 bag record /scan /camera/image_raw /imu /odom -o sensor_data

# Play back recorded data
ros2 bag play sensor_data
```

### Execution

```bash
colcon build --packages-select lab4_sensor_integration
source install/setup.bash
ros2 launch lab4_sensor_integration sensor_integration.launch.py
ros2 run lab4_sensor_integration robot_controller
ros2 run lab4_sensor_integration record_sensor_data
```

### Expected Outcomes

- Lidar sensor publishing scan data
- Camera sensor publishing image data
- IMU sensor publishing orientation data
- Robot performing obstacle avoidance
- Data recorded using ros2 bag

## Acceptance Criteria Met

- [X] Lab 3: Robot model creation and spawning
- [X] Lab 4: Sensor integration and data acquisition
