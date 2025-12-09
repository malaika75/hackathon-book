# Lab 2: ROS 2 Service & Action

## Objective

In this lab, you will create a service for robot state query and an action for a simple navigation task (e.g., move to a target distance) using ROS 2.

## Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Basic understanding of ROS 2 services and actions
- Completion of Lab 1 (or understanding of nodes and topics)

## Lab Setup

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab2_services_actions --dependencies rclpy std_msgs geometry_msgs action_msgs
cd ~/ros2_labs
colcon build --packages-select lab2_services_actions
source install/setup.bash
```

## Lab Implementation

### Task 1: Create Robot State Service

First, let's define a custom service message. Create the service definition file in `lab2_services_actions/srv/RobotState.srv`:

```
# Request - no parameters needed
---
# Response
string robot_name
float64 battery_level
bool is_charging
float64[] joint_positions
int32 error_code
string error_message
```

Now create the service server in `lab2_services_actions/lab2_services_actions/robot_state_server.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from lab2_services_actions.srv import RobotState
import random

class RobotStateService(Node):
    def __init__(self):
        super().__init__('robot_state_service')
        self.srv = self.create_service(RobotState, 'get_robot_state', self.get_robot_state_callback)
        self.get_logger().info('Robot state service server started')

    def get_robot_state_callback(self, request, response):
        # Simulate getting robot state (in real robot, this would query actual sensors)
        response.robot_name = "LabRobot"
        response.battery_level = random.uniform(20.0, 100.0)
        response.is_charging = False
        response.joint_positions = [random.uniform(-1.57, 1.57) for _ in range(6)]  # 6 joints
        response.error_code = 0
        response.error_message = "OK"

        self.get_logger().info(f'Service called, returning state for {response.robot_name}')
        return response

def main(args=None):
    rclpy.init(args=args)
    robot_state_service = RobotStateService()

    try:
        rclpy.spin(robot_state_service)
    except KeyboardInterrupt:
        pass
    finally:
        robot_state_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Create the service client in `lab2_services_actions/lab2_services_actions/robot_state_client.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from lab2_services_actions.srv import RobotState

class RobotStateClient(Node):
    def __init__(self):
        super().__init__('robot_state_client')
        self.cli = self.create_client(RobotState, 'get_robot_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = RobotState.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    robot_state_client = RobotStateClient()

    response = robot_state_client.send_request()
    if response:
        robot_state_client.get_logger().info(
            f'Robot: {response.robot_name}\n'
            f'Battery: {response.battery_level:.2f}%\n'
            f'Charging: {response.is_charging}\n'
            f'Joint positions: {[f"{pos:.2f}" for pos in response.joint_positions]}\n'
            f'Error: {response.error_message} (code: {response.error_code})'
        )
    else:
        robot_state_client.get_logger().info('Service call failed')

    robot_state_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 2: Create Navigation Action

First, define a custom action message. Create the action definition file in `lab2_services_actions/action/MoveToGoal.action`:

```
# Goal
float64 target_distance
float64 max_speed
---
# Result
bool success
float64 actual_distance
string message
---
# Feedback
float64 current_distance
float64 progress_percentage
string status
```

Now create the action server in `lab2_services_actions/lab2_services_actions/navigation_action_server.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from lab2_services_actions.action import MoveToGoal
import time
import math

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        self._action_server = ActionServer(
            self,
            MoveToGoal,
            'move_to_goal',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info('Navigation action server started')

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Get the goal request
        target_distance = goal_handle.request.target_distance
        max_speed = goal_handle.request.max_speed

        # Start executing the action
        feedback_msg = MoveToGoal.Feedback()
        result = MoveToGoal.Result()

        # Simulate movement to target
        current_distance = 0.0
        step_size = max_speed * 0.1  # 0.1 second updates

        while current_distance < target_distance:
            # Check if there was a cancel request
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.actual_distance = current_distance
                result.message = 'Goal canceled'
                self.get_logger().info('Goal canceled')
                return result

            # Update distance
            current_distance = min(current_distance + step_size, target_distance)
            progress = (current_distance / target_distance) * 100.0

            # Publish feedback
            feedback_msg.current_distance = current_distance
            feedback_msg.progress_percentage = progress
            feedback_msg.status = f'Moving: {progress:.1f}% complete'
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(f'Feedback: {feedback_msg.status}')

            # Sleep to simulate movement time
            time.sleep(0.1)

        # Check if goal was canceled
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.actual_distance = current_distance
            result.message = 'Goal canceled'
            return result

        # Goal completed successfully
        goal_handle.succeed()
        result.success = True
        result.actual_distance = current_distance
        result.message = 'Navigation completed successfully'

        self.get_logger().info(f'Navigation completed: {result.message}')
        return result

def main(args=None):
    rclpy.init(args=args)
    navigation_action_server = NavigationActionServer()

    try:
        rclpy.spin(navigation_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Create the action client in `lab2_services_actions/lab2_services_actions/navigation_action_client.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from lab2_services_actions.action import MoveToGoal

class NavigationActionClient(Node):
    def __init__(self):
        super().__init__('navigation_action_client')
        self._action_client = ActionClient(self, MoveToGoal, 'move_to_goal')

    def send_goal(self, target_distance=1.0, max_speed=0.5):
        goal_msg = MoveToGoal.Goal()
        goal_msg.target_distance = target_distance
        goal_msg.max_speed = max_speed

        self.get_logger().info(f'Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending goal: move {target_distance}m at max speed {max_speed}m/s')

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Received feedback: {feedback.status}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = NavigationActionClient()

    # Send a goal to move 2 meters at 0.3 m/s
    action_client.send_goal(target_distance=2.0, max_speed=0.3)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()

if __name__ == '__main__':
    main()
```

### Task 3: Create a Launch File

Create the launch file in `lab2_services_actions/launch/lab2_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Robot State Service Server
        Node(
            package='lab2_services_actions',
            executable='robot_state_server',
            name='robot_state_service',
            output='screen'
        ),

        # Navigation Action Server
        Node(
            package='lab2_services_actions',
            executable='navigation_action_server',
            name='navigation_action_server',
            output='screen'
        )
    ])
```

## Lab Execution

### Building and Running

1. Build the package:
   ```bash
   cd ~/ros2_labs
   colcon build --packages-select lab2_services_actions
   source install/setup.bash
   ```

2. Run the launch file to start servers:
   ```bash
   ros2 launch lab2_services_actions lab2_launch.py
   ```

3. In a new terminal, run the clients:
   ```bash
   # Test the robot state service
   ros2 run lab2_services_actions robot_state_client

   # Test the navigation action
   ros2 run lab2_services_actions navigation_action_client
   ```

### Testing Services and Actions

Open additional terminals to test:

```bash
# List available services
ros2 service list

# Call the robot state service directly
ros2 service call /get_robot_state lab2_services_actions/srv/RobotState

# List available actions
ros2 action list

# Send a goal to the navigation action
ros2 action send_goal /move_to_goal lab2_services_actions/action/MoveToGoal "{target_distance: 3.0, max_speed: 0.5}"
```

## Expected Outcomes

- Robot state service successfully responding to queries
- Navigation action completing goals with proper feedback
- Understanding of service vs action use cases
- Proper implementation of both synchronous (service) and asynchronous (action) communication patterns

## Troubleshooting

1. **Service/action not found**: Make sure both server and client are running
2. **Message type errors**: Verify that service/action definition files are properly defined
3. **Build errors**: Ensure that the `action_msgs` dependency is properly declared

## Solution Guide for Instructors

### Key Learning Points
- Difference between services (synchronous) and actions (asynchronous with feedback)
- When to use each communication pattern
- Implementation of custom service and action message types
- Proper error handling and cancellation in actions

### Assessment Criteria
- Correct implementation of service server and client
- Proper action server with feedback and cancellation handling
- Understanding of when to use services vs actions
- Successful communication patterns demonstrated

## Acceptance Criteria Met

- [X] Complete Lab 2 instructions with expected outcomes
- [X] Solution guides for instructors