# ROS 2 Client Libraries (rclpy/rclcpp)

## Introduction

ROS 2 client libraries provide the APIs that allow you to write ROS 2 programs in different programming languages. The two primary client libraries are rclpy for Python and rclcpp for C++.

## rclpy (Python Client Library)

rclpy is the Python client library for ROS 2. It provides a Pythonic interface to the ROS 2 ecosystem.

### Basic Node Example in Python

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

### Key Features of rclpy

- Simple and intuitive API
- Automatic garbage collection
- Integration with Python's asyncio
- Easy debugging and introspection

## rclcpp (C++ Client Library)

rclcpp is the C++ client library for ROS 2. It provides high-performance access to ROS 2 features.

### Basic Node Example in C++

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

## Python vs C++ Comparison

### When to Use Python (rclpy)

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

### When to Use C++ (rclcpp)

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

## Best Practices

### For Python (rclpy)
- Use type hints for better code documentation
- Handle exceptions appropriately
- Use async/await for non-blocking operations when needed
- Follow PEP 8 style guidelines

### For C++ (rclcpp)
- Use smart pointers for memory management
- Follow RAII principles
- Use const correctness
- Follow ROS 2 C++ style guidelines

## Common Patterns

### Publisher/Subscriber Pattern

Both client libraries support the standard publisher/subscriber pattern:

Python:
```python
# Publisher
publisher = node.create_publisher(MsgType, 'topic_name', qos_profile)

# Subscriber
subscriber = node.create_subscription(MsgType, 'topic_name', callback, qos_profile)
```

C++:
```cpp
// Publisher
auto publisher = this->create_publisher<MsgType>('topic_name', qos_profile);

// Subscriber
auto subscription = this->create_subscription<MsgType>(
    'topic_name', qos_profile, callback);
```

## Acceptance Criteria Met

- [X] Basic node implementation examples in both languages
- [X] Comparison of Python vs C++ use cases
- [X] Best practices for each language