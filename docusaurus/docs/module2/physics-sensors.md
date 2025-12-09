# Physics and Sensor Simulation

## Overview

Physics simulation and sensor modeling are critical components of realistic robotic simulation. This section covers how to configure physics engines and simulate various sensors in Gazebo and other simulation environments.

## Physics Engines

### Available Physics Engines

Gazebo supports multiple physics engines, each with different characteristics:

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

#### Time Step Configuration
- **max_step_size**: Simulation time step (smaller = more accurate but slower)
- **real_time_factor**: Target simulation speed relative to real time
- **real_time_update_rate**: Updates per second

#### Accuracy vs Performance Trade-offs
- Smaller time steps: More accurate but slower
- Higher update rates: More responsive but more CPU intensive
- Adjust based on your application requirements

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
        <remapping>~/camera_info:=/camera/camera_info</remapping>
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
    <pose>0 0 0 0 0 0</pose>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
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
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
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
        <namespace>/my_robot</namespace>
        <remapping>~/out:=/imu</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

### Sensor Noise and Accuracy

#### Adding Realistic Noise

Real sensors have noise characteristics that should be modeled in simulation:

```xml
<camera>
  <!-- Add Gaussian noise to camera -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.007</stddev>
  </noise>
</camera>

<ray>  <!-- LiDAR -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>
  </noise>
</ray>
```

#### Accuracy Considerations

- **Range Sensors**: Consider minimum/maximum range, angular resolution
- **Camera Sensors**: Consider field of view, resolution, distortion
- **IMU Sensors**: Consider bias, drift, noise characteristics

## Example: Complete Sensor Configuration

Here's a complete example of a robot with multiple sensors:

```xml
<?xml version="1.0"?>
<robot name="sensor_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

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
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- LiDAR -->
  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
    </visual>
  </link>
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.1 0 0.15" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins for sensors -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="narrow_stereo_l">
        <horizontal_fov>1.3962634</horizontal_fov>
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
        </ros>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="lidar_link">
    <sensor name="laser" type="ray">
      <pose>0 0 0 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
        <ros>
          <namespace>/sensor_robot</namespace>
          <remapping>~/out:=/scan</remapping>
        </ros>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

## Accuracy Considerations

### Physics Accuracy

- **Contact Stiffness**: Higher values for more rigid contacts, lower for softer
- **Friction Parameters**: Tune for realistic interaction with surfaces
- **Damping**: Helps stabilize simulation and model real-world energy loss

### Sensor Accuracy

- **Model Limitations**: Understand what your sensor model can and cannot simulate
- **Environmental Factors**: Consider lighting, weather, and other environmental effects
- **Calibration**: Simulate sensor calibration procedures

## Troubleshooting Common Issues

### Physics Issues
1. **Objects falling through surfaces**: Check collision geometries and physics parameters
2. **Unstable simulation**: Reduce time step or adjust solver parameters
3. **Jittery movement**: Check mass properties and joint constraints

### Sensor Issues
1. **No sensor data**: Check plugin loading and topic names
2. **Incorrect data**: Verify sensor parameters and coordinate frames
3. **Performance issues**: Reduce sensor update rates or simplify models

## Best Practices

### For Physics Simulation
- Start with default parameters and adjust as needed
- Use realistic mass and inertia values
- Test with different physics engines for your specific application
- Monitor simulation real-time factor to ensure performance

### For Sensor Simulation
- Add realistic noise models to match real sensors
- Validate sensor data against real hardware when possible
- Use appropriate update rates for your application
- Consider computational cost of complex sensor models

## Acceptance Criteria Met

- [X] Physics engine parameter explanations
- [X] Sensor simulation examples (lidar, camera, IMU)
- [X] Accuracy considerations documented