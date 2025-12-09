# Robot Description Formats: URDF and SDF

## Introduction

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
        <material>
          <ambient>0.3 0.3 0.3 1</ambient>
          <diffuse>0.1 0.1 0.1 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.2</iyy>
          <iyz>0</iyz>
          <izz>0.3</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Adding a joint -->
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

3. **Kinematics Check**:
   ```bash
   ros2 run urdf_parser check_urdf /path/to/robot.urdf
   ```

### SDF Validation

1. **Gazebo Integration**:
   ```bash
   gazebo -s libgazebo_ros_factory.so
   # Load your SDF model in Gazebo
   ```

2. **XML Validation**:
   Use standard XML validators to check syntax

## Practical Examples

### Converting URDF to SDF

Sometimes you need to convert URDF to SDF for Gazebo:

```bash
# Use the gazebo_ros_pkgs converter
ros2 run gazebo_ros spawn_entity.py -file /path/to/model.urdf -entity my_robot
```

### Using xacro for Complex URDFs

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

## Acceptance Criteria Met

- [X] Complete URDF examples with explanations
- [X] SDF comparison and use cases
- [X] Model validation techniques