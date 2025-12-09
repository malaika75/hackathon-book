# ROS 2 Launch Files and System Orchestration

## Introduction

Launch files in ROS 2 allow you to start multiple nodes with specific configurations simultaneously. They provide a powerful way to orchestrate complex robotic systems and manage parameters.

## Launch File Basics

### Python Launch Files

ROS 2 uses Python files with a `.py` extension for launch files. Here's a basic example:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop',
            remappings=[
                ('/turtle1/cmd_vel', '/my_turtle/cmd_vel'),
            ]
        )
    ])
```

### Launch File Structure

A launch file typically includes:

1. **Imports**: Import necessary modules from `launch` and `launch_ros`
2. **Launch Description**: The main function that returns a `LaunchDescription`
3. **Node Actions**: Define the nodes to be launched
4. **Additional Actions**: Parameters, timers, conditions, etc.

## Advanced Launch File Features

### Parameters

Launch files can set parameters for nodes:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    param_file = LaunchConfiguration('param_file')

    declare_param_file_cmd = DeclareLaunchArgument(
        'param_file',
        default_value='/path/to/params.yaml',
        description='Path to parameter file'
    )

    # Node with parameters
    controller_node = Node(
        package='my_package',
        executable='controller_node',
        parameters=[
            param_file,  # Load from YAML file
            {'param1': 'value1'},  # Direct parameter assignment
        ]
    )

    return LaunchDescription([
        declare_param_file_cmd,
        controller_node
    ])
```

### YAML Launch Files (Alternative)

ROS 2 also supports YAML-based launch files:

```yaml
launch:
  - node:
      pkg: "turtlesim"
      exec: "turtlesim_node"
      name: "sim"
  - node:
      pkg: "turtlesim"
      exec: "turtle_teleop_key"
      name: "teleop"
      remappings:
        - ["turtle1/cmd_vel", "my_turtle/cmd_vel"]
```

### Conditions and Logic

Launch files can include conditional logic:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    # Conditional node launch
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(use_sim_time)  # Only launch if use_sim_time is true
    )

    return LaunchDescription([
        declare_use_sim_time,
        rviz_node
    ])
```

## Parameter Management Techniques

### YAML Parameter Files

Create parameter files in YAML format:

```yaml
# params.yaml
my_robot_controller:
  ros__parameters:
    kp: 1.0
    ki: 0.1
    kd: 0.05
    max_velocity: 1.0
    wheel_radius: 0.05
```

### Loading Parameters in Launch Files

```python
Node(
    package='my_package',
    executable='controller',
    parameters=[
        'config/params.yaml',  # Load from file
        {'param_name': 'param_value'},  # Direct assignment
        {'use_sim_time': LaunchConfiguration('use_sim_time')},  # From launch arg
    ]
)
```

## ros2 bag Usage for Data Recording

### Recording Data

Use launch files to coordinate data recording:

```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Launch your nodes
    robot_node = Node(
        package='my_robot',
        executable='robot_driver'
    )

    # Record all topics
    record_all = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a'],
        output='screen'
    )

    # Record specific topics
    record_specific = ExecuteProcess(
        cmd=['ros2', 'bag', 'record',
             '/robot/joint_states',
             '/robot/odom',
             '/robot/cmd_vel'],
        output='screen'
    )

    return LaunchDescription([
        robot_node,
        record_specific  # Record specific topics
    ])
```

### Playing Back Data

```python
playback = ExecuteProcess(
    cmd=['ros2', 'bag', 'play', '/path/to/bag/file'],
    output='screen'
)
```

## Best Practices

### Organizing Launch Files

- Keep launch files in the `launch/` directory of your package
- Use descriptive names for launch files
- Separate different configurations into different launch files
- Use launch arguments to make launch files flexible

### Parameter Management

- Use YAML files for complex parameter sets
- Use launch arguments for configurable parameters
- Document parameter meanings and valid ranges
- Group related parameters logically

### System Orchestration

- Start dependencies first (e.g., robot state publisher before nodes that need transforms)
- Use appropriate QoS settings for your application
- Include error handling and logging
- Consider using lifecycle nodes for complex startup sequences

## Common Launch File Patterns

### Robot Bringup

```python
def generate_launch_description():
    # Robot description (URDF)
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('my_robot_description'),
            '/launch/robot_description.launch.py'])
    )

    # Controllers
    controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('my_robot_control'),
            '/launch/controllers.launch.py'])
    )

    # Navigation
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('my_robot_navigation'),
            '/launch/navigation.launch.py'])
    )

    return LaunchDescription([
        robot_description,
        controllers,
        navigation
    ])
```

## Acceptance Criteria Met

- [X] Complete launch file examples
- [X] Parameter management techniques
- [X] ros2 bag usage for data recording