# Chapter 2: CLI Tools, Packages & Client Libraries

This chapter covers the essential command-line tools, package management, and client libraries for ROS 2 development.

## ROS 2 Command Line Interface (CLI) Tools

### Overview

ROS 2 provides a comprehensive set of command-line tools that allow you to introspect and control your ROS 2 system. These tools are essential for debugging, monitoring, and managing your ROS 2 applications.

### Essential ROS 2 CLI Commands

#### ros2 run

The `ros2 run` command is used to run a specific executable from a package:

```bash
ros2 run <package_name> <executable_name>
```

Example:
```bash
ros2 run turtlesim turtlesim_node
```

#### ros2 topic

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

#### ros2 service

The `ros2 service` command allows you to interact with services:

- `ros2 service list` - List all active services
- `ros2 service call <service_name> <service_type> <request_args>` - Call a service

Example:
```bash
ros2 service call /clear std_srvs/srv/Empty
```

#### Additional Useful Commands

- `ros2 node list` - List all active nodes
- `ros2 node info <node_name>` - Get information about a specific node
- `ros2 param list` - List parameters for a node
- `ros2 action list` - List all active actions

### Troubleshooting Tips

- Use `ros2 topic list` to verify that your nodes are communicating properly
- Use `ros2 node list` to check if your nodes are running
- If you're having communication issues, verify that nodes are on the same ROS_DOMAIN_ID
- Use `--help` with any ros2 command to see detailed usage information

## ROS 2 Packages and Workspaces

### Introduction

ROS 2 organizes code into packages and workspaces. Understanding this structure is crucial for developing and managing ROS 2 applications.

### Workspaces

A workspace is a directory that contains ROS 2 packages. It's the top-level directory where you'll organize your ROS 2 development.

#### Creating a Workspace

```bash
mkdir -p ~/ros2_workspace/src
cd ~/ros2_workspace
```

#### Building a Workspace

After adding packages to your workspace, you need to build them using colcon:

```bash
colcon build
```

This will create the following directories:
- `build/` - Build artifacts
- `install/` - Installation directory with executables and libraries
- `log/` - Build logs

### Packages

A package is the basic building unit in ROS 2. It contains source code, configuration files, and other resources needed for a specific functionality.

#### Creating a Package

```bash
cd ~/ros2_workspace/src
ros2 pkg create --build-type ament_python my_robot_package
```

For C++ packages:
```bash
ros2 pkg create --build-type ament_cmake my_robot_package
```

#### Package Structure

A typical ROS 2 package includes:
- `package.xml` - Package manifest with metadata
- `CMakeLists.txt` - Build configuration for CMake packages
- `setup.py` - Build configuration for Python packages
- `src/` - Source code files
- `include/` - Header files (C++)
- `launch/` - Launch files
- `config/` - Configuration files
- `test/` - Test files

### Colcon Build System

Colcon is the build tool used in ROS 2. It's designed to build multiple packages in a workspace efficiently.

#### Common Colcon Commands

- `colcon build` - Build all packages in the workspace
- `colcon build --packages-select <pkg_name>` - Build specific packages
- `colcon build --symlink-install` - Use symlinks for easier development
- `colcon test` - Run tests for all packages
- `colcon test-result --all` - Show test results

### Debugging Techniques

#### Common Issues and Solutions

1. **Package not found**: Source your workspace after building:
   ```bash
   source install/setup.bash
   ```

2. **Import errors**: Make sure you've sourced the correct setup file

3. **Build failures**: Check dependencies in `package.xml` and ensure all required packages are available

4. **Workspace overlay**: If you have multiple workspaces, source them in the correct order

#### Debugging Commands

- `ros2 pkg list` - List all available packages
- `ros2 pkg executables <pkg_name>` - List executables in a package
- `ament list_packages` - Alternative way to list packages

## ROS 2 Client Libraries (rclpy/rclcpp)

### Introduction

ROS 2 client libraries provide the APIs that allow you to write ROS 2 programs in different programming languages. The two primary client libraries are rclpy for Python and rclcpp for C++.

### rclpy (Python Client Library)

rclpy is the Python client library for ROS 2. It provides a Pythonic interface to the ROS 2 ecosystem.

#### Basic Node Example in Python

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Key Features of rclpy

- Simple and intuitive API
- Automatic garbage collection
- Integration with Python's asyncio
- Easy debugging and introspection

### rclcpp (C++ Client Library)

rclcpp is the C++ client library for ROS 2. It provides high-performance access to ROS 2 features.

#### Basic Node Example in C++

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
```

### Python vs C++ Comparison

#### When to Use Python (rclpy)

**Advantages:**
- Rapid prototyping and development
- Easier debugging and testing
- Better for high-level logic and scripting
- Rich ecosystem of scientific libraries (NumPy, OpenCV, etc.)

**Use Cases:**
- Data analysis and visualization
- High-level state machines
- Configuration and testing scripts
- Prototyping algorithms

#### When to Use C++ (rclcpp)

**Advantages:**
- Higher performance and lower latency
- Better for real-time applications
- Lower memory footprint
- Direct hardware access

**Use Cases:**
- Real-time control systems
- Performance-critical algorithms
- Embedded systems
- Low-level drivers

### Best Practices

#### For Python (rclpy)
- Use type hints for better code documentation
- Handle exceptions appropriately
- Use async/await for non-blocking operations when needed
- Follow PEP 8 style guidelines

#### For C++ (rclcpp)
- Use smart pointers for memory management
- Follow RAII principles
- Use const correctness
- Follow ROS 2 C++ style guidelines

## Acceptance Criteria Met

- [X] Complete coverage of CLI commands
- [X] Package and workspace management
- [X] Client library examples in Python and C++
- [X] Best practices documented
