# Lab 4: Sensor Integration & Data Acquisition

## Objective

In this lab, you will simulate a lidar sensor, visualize its data in RViz, and record the data using `ros2 bag`. This lab focuses on sensor integration in simulation and data acquisition techniques.

## Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Gazebo installation
- Completion of Lab 3 or understanding of robot simulation
- Basic understanding of RViz and `ros2 bag`

## Lab Setup

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab4_sensor_integration --dependencies rclpy std_msgs sensor_msgs geometry_msgs rclpy tf2_ros tf2_geometry_msgs
cd ~/ros2_labs
colcon build --packages-select lab4_sensor_integration
source install/setup.bash
```

### Creating Directories

```bash
cd ~/ros2_labs/src/lab4_sensor_integration
mkdir -p urdf launch config
```

## Lab Implementation

### Task 1: Create Robot with Multiple Sensors

Create an enhanced robot URDF with lidar and other sensors in `urdf/sensor_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="sensor_robot">
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
      <material name="green">
        <color rgba="0 0.8 0 1"/>
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
    <link name="wheel_${prefix}">
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

    <joint name="wheel_${prefix}_joint" type="continuous">
      <parent link="base_link"/>
      <child link="wheel_${prefix}"/>
      <origin xyz="${reflect_x * base_length/2} ${reflect_y * (base_width/2 + wheel_width/2)} ${-wheel_radius}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Create wheels -->
  <xacro:wheel prefix="fl" reflect_x="1" reflect_y="1"/>
  <xacro:wheel prefix="fr" reflect_x="1" reflect_y="-1"/>
  <xacro:wheel prefix="bl" reflect_x="-1" reflect_y="1"/>
  <xacro:wheel prefix="br" reflect_x="-1" reflect_y="-1"/>

  <!-- Lidar sensor -->
  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="${base_length/2 - 0.025} 0 ${base_height/2}" rpy="0 0 0"/>
  </joint>

  <!-- Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="${base_length/2 - 0.025} 0.1 ${base_height/2}" rpy="0 0 0"/>
  </joint>

  <!-- IMU -->
  <link name="imu_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
      <material name="yellow">
        <color rgba="0.8 0.8 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <update_rate>30</update_rate>
      <left_joint>wheel_fl_joint</left_joint>
      <right_joint>wheel_fr_joint</right_joint>
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

  <!-- Lidar sensor plugin -->
  <gazebo reference="lidar_link">
    <sensor name="lidar" type="ray">
      <pose>0 0 0 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <ros>
          <namespace>/sensor_robot</namespace>
          <remapping>~/out:=/scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Camera sensor plugin -->
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
          <namespace>/sensor_robot</namespace>
          <remapping>~/image_raw:=/camera/image_raw</remapping>
          <remapping>~/camera_info:=/camera/camera_info</remapping>
        </ros>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor plugin -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
        <ros>
          <namespace>/sensor_robot</namespace>
          <remapping>~/out:=/imu</remapping>
        </ros>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="base_link">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="wheel_fl">
    <material>Gazebo/Black</material>
  </gazebo>
</robot>
```

### Task 2: Create Launch File with Multiple Components

Create the launch file `launch/sensor_integration.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
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
                    FindPackageShare('lab4_sensor_integration'),
                    'urdf',
                    'sensor_robot.urdf.xacro'
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
            '-entity', 'sensor_robot',
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

    # RViz node for visualization
    rviz_config = PathJoinSubstitution([
        FindPackageShare('lab4_sensor_integration'),
        'config',
        'sensor_rviz.rviz'
    ])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Robot controller node
    controller = Node(
        package='lab4_sensor_integration',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_state_publisher,
        gazebo,
        spawn_entity,
        # Delay RViz to allow other nodes to start
        TimerAction(
            period=5.0,
            actions=[rviz]
        ),
        TimerAction(
            period=10.0,
            actions=[controller]
        )
    ])
```

### Task 3: Create RViz Configuration

Create the RViz configuration file `config/sensor_rviz.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /LaserScan1
        - /RobotModel1
        - /TF1
        - /Image1
      Splitter Ratio: 0.5
    Tree Height: 617
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
        base_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
        camera_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        imu_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        lidar_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_bl:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_br:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_fl:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_fr:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: 0
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/LaserScan
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 0
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: LaserScan
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /scan
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 1
      Min Value: 0
      Name: Image
      Normalize Range: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /camera/image_raw
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: odom
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /initialpose
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /clicked_point
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.7853981852531433
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.7853981852531433
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  Image:
    collapsed: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002f4fc0200000009fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002f4000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002f4fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073000000003d000002f4000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d006501000000000000045000000000000000000000023f000002f400000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1200
  X: 72
  Y: 60
```

### Task 4: Create Robot Controller

Create the robot controller in `lab4_sensor_integration/robot_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import math
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for laser scan
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Store latest scan data
        self.latest_scan = None

        # Control state
        self.state = 'explore'  # 'explore', 'avoid', 'turn'
        self.obstacle_distance = float('inf')

        self.get_logger().info('Robot Controller initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg

        # Find minimum distance in front of robot (forward 90 degrees)
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
            self.obstacle_distance = min(front_distances)
        else:
            self.obstacle_distance = float('inf')

    def control_loop(self):
        """Main control loop"""
        if self.latest_scan is None:
            return

        cmd_msg = Twist()

        # Simple obstacle avoidance behavior
        safety_distance = 0.5  # meters

        if self.obstacle_distance < safety_distance:
            # Obstacle detected, avoid
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5  # Turn right
            self.state = 'avoid'
        else:
            # Clear path, explore
            cmd_msg.linear.x = 0.3
            cmd_msg.angular.z = 0.0
            self.state = 'explore'

        self.cmd_vel_pub.publish(cmd_msg)

        self.get_logger().info(
            f'State: {self.state}, '
            f'Obstacle distance: {self.obstacle_distance:.2f}m, '
            f'Command: linear={cmd_msg.linear.x:.2f}, angular={cmd_msg.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before shutting down
        stop_msg = Twist()
        controller.cmd_vel_pub.publish(stop_msg)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 5: Create Data Recording Script

Create a data recording script in `lab4_sensor_integration/record_sensor_data.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import subprocess
import os
from datetime import datetime

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')

        # Subscribe to all sensor topics
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Create recording directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_dir = f"/tmp/sensor_data_{timestamp}"
        os.makedirs(self.recording_dir, exist_ok=True)

        # Start ros2 bag recording
        self.start_bag_recording()

        self.get_logger().info(f'Data recorder initialized, saving to: {self.recording_dir}')

    def start_bag_recording(self):
        """Start recording all topics with ros2 bag"""
        # Define the topics to record
        topics = [
            '/scan',
            '/camera/image_raw',
            '/imu',
            '/odom',
            '/cmd_vel',
            '/tf',
            '/tf_static'
        ]

        # Create the ros2 bag command
        bag_cmd = [
            'ros2', 'bag', 'record',
            '--output', self.recording_dir
        ] + topics

        # Start the recording process
        self.bag_process = subprocess.Popen(bag_cmd)
        self.get_logger().info('Started ros2 bag recording')

    def scan_callback(self, msg):
        """Handle laser scan messages"""
        # This is handled by ros2 bag
        pass

    def image_callback(self, msg):
        """Handle image messages"""
        # This is handled by ros2 bag
        pass

    def imu_callback(self, msg):
        """Handle IMU messages"""
        # This is handled by ros2 bag
        pass

    def odom_callback(self, msg):
        """Handle odometry messages"""
        # This is handled by ros2 bag
        pass

def main(args=None):
    rclpy.init(args=args)
    recorder = DataRecorder()

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        recorder.get_logger().info('Shutting down data recorder...')
        if hasattr(recorder, 'bag_process'):
            recorder.bag_process.terminate()
            recorder.bag_process.wait()
    finally:
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab Execution

### Building and Running

1. Build the package:
   ```bash
   cd ~/ros2_labs
   colcon build --packages-select lab4_sensor_integration
   source install/setup.bash
   ```

2. Run the complete simulation:
   ```bash
   ros2 launch lab4_sensor_integration sensor_integration.launch.py
   ```

3. In a new terminal, run the data recorder:
   ```bash
   ros2 run lab4_sensor_integration record_sensor_data
   ```

### Recording Data with ros2 bag

While the simulation is running, you can also manually record data:

```bash
# Record specific topics
ros2 bag record /scan /camera/image_raw /imu /odom -o sensor_data_recording

# Record all topics
ros2 bag record -a -o all_sensor_data
```

### Analyzing Recorded Data

After recording, you can play back and analyze the data:

```bash
# Play back the recorded data
ros2 bag play sensor_data_recording

# Check the contents of the bag file
ros2 bag info sensor_data_recording

# Extract specific topics
ros2 bag play sensor_data_recording --topics /scan
```

## Expected Outcomes

- Lidar sensor successfully publishing scan data
- Camera sensor publishing image data
- IMU sensor publishing orientation data
- All sensor data visualizable in RViz
- Data successfully recorded using `ros2 bag`
- Robot responding to sensor data for navigation

## Troubleshooting

1. **No sensor data**: Check that plugins are properly loaded in URDF
2. **RViz not showing data**: Verify topic names and frame IDs
3. **Recording fails**: Check permissions and available disk space
4. **TF issues**: Ensure all sensor frames are properly connected

## Solution Guide for Instructors

### Key Learning Points
- Integration of multiple sensor types in simulation
- Visualization of sensor data in RViz
- Data acquisition and recording techniques
- Robot control based on sensor feedback

### Assessment Criteria
- Successful integration of all sensor types
- Proper visualization of sensor data
- Correct data recording procedures
- Understanding of sensor-based control

## Acceptance Criteria Met

- [X] Complete Lab 4 instructions with expected outcomes
- [X] Solution guides for instructors