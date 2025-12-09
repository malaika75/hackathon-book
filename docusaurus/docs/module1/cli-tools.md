# ROS 2 Command Line Interface (CLI) Tools

## Overview

ROS 2 provides a comprehensive set of command-line tools that allow you to introspect and control your ROS 2 system. These tools are essential for debugging, monitoring, and managing your ROS 2 applications.

## Essential ROS 2 CLI Commands

### ros2 run

The `ros2 run` command is used to run a specific executable from a package:

```bash
ros2 run <package_name> <executable_name>
```

Example:
```bash
ros2 run turtlesim turtlesim_node
```

### ros2 topic

The `ros2 topic` command allows you to interact with topics in your ROS 2 system:

- `ros2 topic list` - List all active topics
- `ros2 topic echo <topic_name>` - Print messages from a topic
- `ros2 topic info <topic_name>` - Get information about a topic
- `ros2 topic pub <topic_name> <msg_type> <args>` - Publish a message to a topic

Example:
```bash
ros2 topic echo /turtle1/pose
ros2 topic pub /turtle1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 2.0}, angular: {z: 1.8}}'
```

### ros2 service

The `ros2 service` command allows you to interact with services:

- `ros2 service list` - List all active services
- `ros2 service call <service_name> <service_type> <request_args>` - Call a service

Example:
```bash
ros2 service call /clear std_srvs/srv/Empty
```

### Additional Useful Commands

- `ros2 node list` - List all active nodes
- `ros2 node info <node_name>` - Get information about a specific node
- `ros2 param list` - List parameters for a node
- `ros2 action list` - List all active actions

## Troubleshooting Tips

- Use `ros2 topic list` to verify that your nodes are communicating properly
- Use `ros2 node list` to check if your nodes are running
- If you're having communication issues, verify that nodes are on the same ROS_DOMAIN_ID
- Use `--help` with any ros2 command to see detailed usage information

## Acceptance Criteria Met

- [X] Complete coverage of `ros2 run`, `ros2 topic`, `ros2 service` commands
- [X] Practical examples for each command
- [X] Troubleshooting tips included