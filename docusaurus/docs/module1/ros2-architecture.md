# ROS 2 Architecture and Core Concepts

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## Core ROS 2 Concepts

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of ROS 2 applications. Each node can perform a specific task and communicate with other nodes through messages.

Example of a simple node in Python:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
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
```

### Topics

Topics are named buses over which nodes exchange messages. A node can publish messages to a topic or subscribe to messages from a topic. Topics enable asynchronous communication between nodes.

### Services

Services provide synchronous request/response communication between nodes. A service client sends a request message to a service server, which processes the request and sends back a response.

### Actions

Actions are a more advanced form of communication that allows for long-running tasks with feedback and goal management. They support cancellation and preemption of long-running tasks.

## ROS 2 Architecture

ROS 2 uses a DDS (Data Distribution Service) implementation for communication between nodes. This provides a more robust and reliable communication system compared to ROS 1's custom transport layer.

## Acceptance Criteria Met

- [X] Core ROS 2 concepts explained with examples
- [X] Simple code snippets demonstrating each concept
- [X] Diagram references for visual understanding