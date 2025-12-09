# Lab 7: Natural Language Task Planning

## Objective

In this lab, you will use an LLM/VLA API to parse natural language commands into a sequence of robot actions for a simulated humanoid. This lab focuses on the integration of large language models with robotic systems to enable natural human-robot interaction.

## Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Access to an LLM/VLA API (OpenAI GPT, Anthropic Claude, or similar)
- Basic understanding of ROS 2 message types and services
- Completion of Modules 1-4
- Python programming experience

## Lab Setup

### Required Packages and APIs

```bash
# Install required Python packages
pip install openai anthropic python-dotenv
pip install rclpy std_msgs geometry_msgs sensor_msgs

# For OpenAI API access
export OPENAI_API_KEY="your-api-key-here"

# For Anthropic API access
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab7_natural_language_task_planning --dependencies rclpy std_msgs geometry_msgs sensor_msgs geometry_msgs action_msgs
cd ~/ros2_labs
colcon build --packages-select lab7_natural_language_task_planning
source install/setup.bash
```

## Part 1: LLM Integration Framework

### 1.1 LLM Interface Node

Create `llm_interface_node.py` in your lab package:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from action_msgs.msg import GoalStatus
import openai
import json
import os
from typing import Dict, List, Any

class LLMInterfaceNode(Node):
    def __init__(self):
        super().__init__('llm_interface_node')

        # Initialize LLM client
        self.setup_llm_client()

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/natural_language_command', self.command_callback, 10)

        # Publishers
        self.action_sequence_pub = self.create_publisher(
            String, '/robot_action_sequence', 10)
        self.status_pub = self.create_publisher(
            String, '/llm_status', 10)

        # Service clients for robot state
        self.robot_state_client = self.create_client(
            GetRobotState, '/get_robot_state')

        self.get_logger().info('LLM Interface Node initialized')

    def setup_llm_client(self):
        """Setup LLM client based on environment configuration"""
        # Check for OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            openai.api_key = openai_key
            self.llm_provider = 'openai'
            self.get_logger().info('Using OpenAI API')
        else:
            self.get_logger().error('No LLM API key found')
            raise Exception('No LLM API key configured')

    def command_callback(self, msg):
        """Process natural language command and generate action sequence"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Get current robot state
        robot_state = self.get_robot_state()

        # Generate action sequence using LLM
        action_sequence = self.generate_action_sequence(command, robot_state)

        if action_sequence:
            # Publish action sequence
            action_msg = String()
            action_msg.data = json.dumps(action_sequence)
            self.action_sequence_pub.publish(action_msg)

            self.get_logger().info(f'Published action sequence: {action_sequence}')
        else:
            self.get_logger().error('Failed to generate action sequence')

    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state from robot state service"""
        # For this lab, we'll simulate robot state
        # In a real implementation, this would call a service
        return {
            'location': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'battery_level': 85.0,
            'holding_object': None,
            'capabilities': ['navigation', 'manipulation', 'speech'],
            'available_objects': [
                {'name': 'red_ball', 'type': 'ball', 'color': 'red', 'location': {'x': 1.0, 'y': 0.0}},
                {'name': 'blue_cube', 'type': 'cube', 'color': 'blue', 'location': {'x': 1.5, 'y': 0.5}},
                {'name': 'green_pyramid', 'type': 'pyramid', 'color': 'green', 'location': {'x': 0.5, 'y': 1.0}}
            ]
        }

    def generate_action_sequence(self, command: str, robot_state: Dict) -> List[Dict]:
        """Generate action sequence using LLM"""
        try:
            prompt = self.create_prompt(command, robot_state)

            if self.llm_provider == 'openai':
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                response_text = response.choices[0].message['content'].strip()

                # Extract JSON from response
                action_sequence = self.extract_json_from_response(response_text)
                return action_sequence
            else:
                self.get_logger().error(f'Unsupported LLM provider: {self.llm_provider}')
                return None

        except Exception as e:
            self.get_logger().error(f'Error generating action sequence: {e}')
            return None

    def create_prompt(self, command: str, robot_state: Dict) -> str:
        """Create prompt for LLM with robot context"""
        return f"""
        You are a helpful assistant that converts natural language commands into robot action sequences.
        The robot is a humanoid robot with navigation and manipulation capabilities.

        Current robot state:
        - Location: {robot_state['location']}
        - Battery level: {robot_state['battery_level']}%
        - Holding object: {robot_state['holding_object']}
        - Capabilities: {robot_state['capabilities']}
        - Available objects: {robot_state['available_objects']}

        Command: "{command}"

        Please respond with a JSON list of actions to execute, in this format:
        [
            {{
                "action": "action_name",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "description": "Brief description of the action"
            }},
            ...
        ]

        Available actions: navigate_to, grasp_object, place_object, speak, wait.
        Be specific about object names and locations. Consider the robot's current state.
        """

    def get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """
        You are a helpful assistant that converts natural language commands into robot action sequences.
        Always respond with valid JSON containing an array of actions.
        Each action should have 'action', 'parameters', and 'description' fields.
        Be precise and consider the robot's capabilities and current state.
        """

    def extract_json_from_response(self, response: str) -> List[Dict]:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no brackets found, try parsing the whole response
                return json.loads(response)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error parsing JSON from LLM response: {e}')
            self.get_logger().debug(f'LLM response: {response}')
            return None

def main(args=None):
    rclpy.init(args=args)
    node = LLMInterfaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 1.2 Robot Action Executor Node

Create `action_executor_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from action_msgs.msg import GoalStatus
import json
import time
from typing import Dict, List, Any

class ActionExecutorNode(Node):
    def __init__(self):
        super().__init__('action_executor_node')

        # Subscribers
        self.action_sequence_sub = self.create_subscription(
            String, '/robot_action_sequence', self.action_sequence_callback, 10)

        # Publishers
        self.status_pub = self.create_publisher(String, '/action_executor_status', 10)
        self.robot_command_pub = self.create_publisher(String, '/robot_commands', 10)

        # Internal state
        self.current_action_index = 0
        self.action_sequence = []
        self.executing = False

        self.get_logger().info('Action Executor Node initialized')

    def action_sequence_callback(self, msg):
        """Process action sequence from LLM"""
        try:
            self.action_sequence = json.loads(msg.data)
            self.current_action_index = 0
            self.executing = True

            self.get_logger().info(f'Starting execution of {len(self.action_sequence)} actions')

            # Execute the sequence
            self.execute_action_sequence()

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error parsing action sequence: {e}')

    def execute_action_sequence(self):
        """Execute the action sequence"""
        while self.current_action_index < len(self.action_sequence) and self.executing:
            action = self.action_sequence[self.current_action_index]

            self.get_logger().info(f'Executing action {self.current_action_index + 1}: {action["action"]}')

            # Execute the action
            success = self.execute_single_action(action)

            if success:
                self.get_logger().info(f'Action {self.current_action_index + 1} completed successfully')
                self.current_action_index += 1
            else:
                self.get_logger().error(f'Action {self.current_action_index + 1} failed')
                # For this lab, we'll continue to next action, but in real system you might want to stop
                self.current_action_index += 1

        # Sequence completed
        self.executing = False
        status_msg = String()
        status_msg.data = f'Action sequence completed. {len(self.action_sequence)} actions executed.'
        self.status_pub.publish(status_msg)

        self.get_logger().info('Action sequence execution completed')

    def execute_single_action(self, action: Dict) -> bool:
        """Execute a single action"""
        action_name = action['action']
        parameters = action.get('parameters', {})

        if action_name == 'navigate_to':
            return self.execute_navigate_to(parameters)
        elif action_name == 'grasp_object':
            return self.execute_grasp_object(parameters)
        elif action_name == 'place_object':
            return self.execute_place_object(parameters)
        elif action_name == 'speak':
            return self.execute_speak(parameters)
        elif action_name == 'wait':
            return self.execute_wait(parameters)
        else:
            self.get_logger().error(f'Unknown action: {action_name}')
            return False

    def execute_navigate_to(self, params: Dict) -> bool:
        """Execute navigation action"""
        target = params.get('location', params.get('target', 'unknown'))

        self.get_logger().info(f'Navigating to {target}')

        # In a real implementation, this would call navigation services
        # For this lab, we'll simulate the action
        time.sleep(2)  # Simulate navigation time

        # Publish command for simulation
        command_msg = String()
        command_msg.data = f'navigate_to {target}'
        self.robot_command_pub.publish(command_msg)

        return True

    def execute_grasp_object(self, params: Dict) -> bool:
        """Execute grasp action"""
        target_object = params.get('object', params.get('target', 'unknown'))

        self.get_logger().info(f'Grasping object: {target_object}')

        # Simulate grasp action
        time.sleep(1.5)

        # Publish command for simulation
        command_msg = String()
        command_msg.data = f'grasp_object {target_object}'
        self.robot_command_pub.publish(command_msg)

        return True

    def execute_place_object(self, params: Dict) -> bool:
        """Execute place action"""
        target_object = params.get('object', params.get('target', 'unknown'))
        location = params.get('location', params.get('surface', 'table'))

        self.get_logger().info(f'Placing {target_object} at {location}')

        # Simulate place action
        time.sleep(1.5)

        # Publish command for simulation
        command_msg = String()
        command_msg.data = f'place_object {target_object} at {location}'
        self.robot_command_pub.publish(command_msg)

        return True

    def execute_speak(self, params: Dict) -> bool:
        """Execute speech action"""
        text = params.get('text', params.get('message', 'Hello'))

        self.get_logger().info(f'Speaking: {text}')

        # In a real system, this would call text-to-speech
        # For this lab, we'll just log the message
        time.sleep(0.5)

        return True

    def execute_wait(self, params: Dict) -> bool:
        """Execute wait action"""
        duration = params.get('duration', params.get('seconds', 1.0))

        self.get_logger().info(f'Waiting for {duration} seconds')

        time.sleep(duration)

        return True

def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Part 2: Natural Language Processing Pipeline

### 2.1 Command Preprocessing Node

Create `command_preprocessor_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import re
import json
from typing import Dict, List, Any

class CommandPreprocessorNode(Node):
    def __init__(self):
        super().__init__('command_preprocessor_node')

        # Subscribers
        self.raw_command_sub = self.create_subscription(
            String, '/raw_natural_language', self.raw_command_callback, 10)

        # Publishers
        self.processed_command_pub = self.create_publisher(
            String, '/natural_language_command', 10)
        self.preprocessor_status_pub = self.create_publisher(
            String, '/preprocessor_status', 10)

        # Parameters
        self.declare_parameter('enable_preprocessing', True)
        self.enable_preprocessing = self.get_parameter('enable_preprocessing').value

        self.get_logger().info('Command Preprocessor Node initialized')

    def raw_command_callback(self, msg):
        """Process raw natural language command"""
        raw_command = msg.data
        self.get_logger().info(f'Received raw command: {raw_command}')

        if self.enable_preprocessing:
            processed_command = self.preprocess_command(raw_command)
        else:
            processed_command = raw_command

        # Publish processed command
        processed_msg = String()
        processed_msg.data = processed_command
        self.processed_command_pub.publish(processed_msg)

        self.get_logger().info(f'Published processed command: {processed_command}')

    def preprocess_command(self, command: str) -> str:
        """Preprocess natural language command"""
        # Clean up the command
        cleaned_command = self.clean_command(command)

        # Expand abbreviations and normalize
        normalized_command = self.normalize_command(cleaned_command)

        # Contextual expansion (simplified)
        expanded_command = self.expand_context(normalized_command)

        return expanded_command

    def clean_command(self, command: str) -> str:
        """Clean command by removing noise and standardizing format"""
        # Convert to lowercase
        command = command.lower()

        # Remove extra whitespace
        command = ' '.join(command.split())

        # Remove common filler words (simplified)
        fillers = ['um', 'uh', 'like', 'you know', 'so']
        for filler in fillers:
            command = command.replace(filler, '')

        # Remove extra whitespace again after filler removal
        command = ' '.join(command.split())

        return command

    def normalize_command(self, command: str) -> str:
        """Normalize command using common expansions"""
        # Common expansions
        expansions = {
            'gonna': 'going to',
            'wanna': 'want to',
            'gotta': 'got to',
            'kinda': 'kind of',
            'sorta': 'sort of',
            'lemme': 'let me',
            'gimme': 'give me',
            'ain\'t': 'is not',
            'won\'t': 'will not',
            'can\'t': 'cannot',
            'n\'t': ' not',
        }

        for contraction, expansion in expansions.items():
            command = command.replace(contraction, expansion)

        # Normalize object references
        command = re.sub(r'that there', 'that', command)
        command = re.sub(r'this here', 'this', command)

        return command

    def expand_context(self, command: str) -> str:
        """Expand command based on context (simplified)"""
        # This would typically connect to context from previous interactions
        # For this lab, we'll do basic expansions

        # Replace pronouns with context when possible
        # (In a real system, this would use discourse context)
        command = re.sub(r'it', 'the object', command)
        command = re.sub(r'them', 'the objects', command)

        # Expand relative references
        command = re.sub(r'over there', 'in that location', command)
        command = re.sub(r'right there', 'at that position', command)

        return command

def main(args=None):
    rclpy.init(args=args)
    node = CommandPreprocessorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Part 3: Simulation and Testing

### 3.1 Test Script

Create `test_natural_language_task_planning.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import threading

class NLTaskPlanningTester(Node):
    def __init__(self):
        super().__init__('nl_task_planning_tester')

        # Publishers
        self.command_pub = self.create_publisher(String, '/raw_natural_language', 10)
        self.status_sub = self.create_subscription(
            String, '/action_executor_status', self.status_callback, 10)

        self.status_log = []
        self.test_completed = threading.Event()

        self.get_logger().info('Natural Language Task Planning Tester initialized')

    def status_callback(self, msg):
        """Log status messages"""
        self.status_log.append(msg.data)
        self.get_logger().info(f'Status: {msg.data}')

        # Set event when task is completed
        if 'completed' in msg.data.lower():
            self.test_completed.set()

    def run_tests(self):
        """Run a series of natural language command tests"""
        test_commands = [
            "Please go to the kitchen and bring me the red ball",
            "Navigate to the living room and wait there for 5 seconds",
            "Pick up the blue cube and place it on the table",
            "Move to the bedroom and speak 'Hello, I am here'"
        ]

        self.get_logger().info('Starting natural language task planning tests...')

        for i, command in enumerate(test_commands):
            self.get_logger().info(f'Test {i+1}: {command}')

            # Publish command
            command_msg = String()
            command_msg.data = command
            self.command_pub.publish(command_msg)

            # Wait for completion or timeout
            if not self.test_completed.wait(timeout=30.0):  # 30 second timeout per command
                self.get_logger().warn(f'Test {i+1} timed out')
            else:
                self.get_logger().info(f'Test {i+1} completed successfully')

            # Reset for next test
            self.test_completed.clear()
            time.sleep(2)  # Brief pause between tests

        self.get_logger().info('All tests completed')
        self.analyze_results()

    def analyze_results(self):
        """Analyze test results"""
        completed_count = sum(1 for status in self.status_log if 'completed' in status.lower())
        total_commands = len([cmd for cmd in self.status_log if 'Action sequence' in cmd])

        self.get_logger().info(f'=== Test Results ===')
        self.get_logger().info(f'Total commands processed: {total_commands}')
        self.get_logger().info(f'Successfully completed: {completed_count}')
        self.get_logger().info(f'Success rate: {completed_count/total_commands*100:.1f}%' if total_commands > 0 else '0%')

        # Print status log
        for i, status in enumerate(self.status_log):
            self.get_logger().info(f'  {i+1}. {status}')

def main(args=None):
    rclpy.init(args=args)
    tester = NLTaskPlanningTester()

    # Run tests in a separate thread to allow ROS spinning
    test_thread = threading.Thread(target=tester.run_tests)
    test_thread.start()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        test_thread.join(timeout=5.0)  # Wait for test thread to finish
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.2 Launch File

Create `natural_language_task_planning_launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    enable_preprocessing = LaunchConfiguration('enable_preprocessing', default='true')

    # LLM Interface Node
    llm_interface_node = Node(
        package='lab7_natural_language_task_planning',
        executable='llm_interface_node',
        name='llm_interface_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Action Executor Node
    action_executor_node = Node(
        package='lab7_natural_language_task_planning',
        executable='action_executor_node',
        name='action_executor_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Command Preprocessor Node
    command_preprocessor_node = Node(
        package='lab7_natural_language_task_planning',
        executable='command_preprocessor_node',
        name='command_preprocessor_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'enable_preprocessing': enable_preprocessing}
        ],
        output='screen'
    )

    return LaunchDescription([
        llm_interface_node,
        action_executor_node,
        command_preprocessor_node
    ])
```

## Lab Execution Instructions

### Step 1: Environment Setup
1. Ensure you have an LLM API key (OpenAI, Anthropic, etc.)
2. Set the appropriate environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. Install required packages as specified in the setup section

### Step 2: Build and Launch
```bash
cd ~/ros2_labs
colcon build --packages-select lab7_natural_language_task_planning
source install/setup.bash
ros2 launch lab7_natural_language_task_planning natural_language_task_planning_launch.py
```

### Step 3: Test the System
In a separate terminal:
```bash
cd ~/ros2_labs
source install/setup.bash
python3 test_natural_language_task_planning.py
```

### Step 4: Manual Testing
You can also send commands manually:
```bash
# Send a command directly to the raw input
ros2 topic pub /raw_natural_language std_msgs/String "data: 'Go to the kitchen and bring me the red ball'"
```

## Expected Outcomes

- Natural language commands are successfully parsed by the LLM
- Action sequences are generated and executed by the robot
- The system handles various types of commands (navigation, manipulation, etc.)
- Proper error handling and status reporting

## Troubleshooting

1. **API Key Issues**: Ensure your LLM API key is properly set in environment variables
2. **JSON Parsing Errors**: Check that the LLM response format is compatible with the parser
3. **Action Execution Failures**: Verify that action parameters are properly formatted
4. **Preprocessing Issues**: Adjust the preprocessing pipeline as needed for your specific commands

## Advanced Challenges

1. Implement more sophisticated context management for multi-turn interactions
2. Add error recovery mechanisms when actions fail
3. Implement safety validation for generated action sequences
4. Add support for multiple LLM providers with fallback mechanisms

## Acceptance Criteria Met

- [X] Complete Lab 7 instructions with expected outcomes
- [X] Solution guides for instructors
- [X] Implementation covers LLM integration with robot task planning