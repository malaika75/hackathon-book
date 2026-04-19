# Chapter 3: Action Generation & Labs

This chapter covers action generation from language and includes hands-on labs for natural language task planning and VLA environment interaction.

---

## Action Generation from Language

### Overview

Action generation translates natural language commands into executable robot actions, bridging abstract language and concrete physical actions.

### Instruction-to-Action Mapping Techniques

#### 1. Template-Based Mapping

Uses predefined templates that map language patterns to actions:

```python
class TemplateBasedActionMapper:
    def __init__(self):
        self.action_templates = [
            {
                'pattern': r'go to (the )?(?P<location>\w+)',
                'action': 'navigate',
                'parameters': ['location']
            },
            {
                'pattern': r'pick up (the )?(?P<object>\w+)',
                'action': 'grasp',
                'parameters': ['object']
            },
            {
                'pattern': r'put (the )?(?P<object>\w+) on (the )?(?P<surface>\w+)',
                'action': 'place',
                'parameters': ['object', 'surface']
            },
            {
                'pattern': r'bring (the )?(?P<object>\w+) to (me|the )(?P<target>\w+)',
                'action': 'fetch_and_deliver',
                'parameters': ['object', 'target']
            }
        ]

    def map_to_action(self, command):
        import re
        for template in self.action_templates:
            match = re.search(template['pattern'], command.lower())
            if match:
                action = template['action']
                parameters = {param: match.group(param) for param in template['parameters']}
                return {'action': action, 'parameters': parameters}
        return None
```

#### 2. LLM-Based Action Generation

Uses large language models to generate actions:

```python
class LLMActionGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_action_plan(self, command, context):
        prompt = f"""Given the command: "{command}"
And the current context: {context}

Break this down into a sequence of robot actions.
Return a JSON list of actions, each with action_type and parameters."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)
```

### Action Execution Pipeline

```
Command → Intent Recognition → Parameter Extraction
    ↓
Task Planning → Action Sequence Generation
    ↓
Motion Planning → Joint Trajectories
    ↓
Robot Execution → Motor Commands
```

---

## Lab 7: Natural Language Task Planning

### Objective

Use an LLM/VLA API to parse natural language commands into a sequence of robot actions.

### Prerequisites

- ROS 2 Humble or later
- Access to LLM/VLA API (OpenAI, Anthropic, etc.)
- Python programming experience

### Step 1: LLM Interface Setup

```python
import rclpy
from rclpy.node import Node
import openai

class LLMTaskPlanner(Node):
    def __init__(self):
        super().__init__('llm_task_planner')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.command_sub = self.create_subscription(String, '/voice_command', self.command_callback, 10)
        self.action_pub = self.create_publisher(ActionSequence, '/action_plan', 10)

    def command_callback(self, msg):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Parse this robot command into actions: {msg.data}"
            }]
        )
        actions = self.parse_llm_response(response.choices[0].message.content)
        self.action_pub.publish(actions)
```

### Step 2: Command Parsing

Test with commands like:
- "Go to the kitchen and bring me a cup"
- "Pick up the book from the table and place it on the shelf"
- "Follow me to the living room"

### Expected Outcomes

- LLM successfully parses commands into action sequences
- Actions are correctly mapped to robot capabilities
- System handles complex multi-step commands

---

## Lab 8: VLA for Environment Interaction

### Objective

Guide a humanoid robot through complex tasks using natural language instructions with visual understanding.

### Prerequisites

- ROS 2 Humble or later
- VLA API access
- Gazebo simulation

### Step 1: Scene Understanding

```python
class SceneUnderstanding(Node):
    def __init__(self):
        super().__init__('scene_understanding')
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.process_image, 10)

    def process_image(self, msg):
        # Get scene description from VLA
        scene_description = self.vla_model.describe_scene(cv_bridge.imgmsg_to_cv2(msg))
        self.get_logger().info(f'Scene: {scene_description}')
```

### Step 2: VLA Integration

```python
class VLAController(Node):
    def __init__(self):
        super().__init__('vla_controller')
        self.vla = VLAInterface()
        self.cmd_sub = self.create_subscription(String, '/vla_command', self.execute_command, 10)

    def execute_command(self, msg):
        # Get current observation
        observation = self.get_observation()

        # Get action from VLA
        action = self.vla.get_action(observation, msg.data)

        # Execute on robot
        self.execute_action(action)
```

### Step 3: Execute Complex Tasks

Test commands like:
- "Find the red cup on the table and bring it to me"
- "Navigate around the obstacle and go to the door"
- "Pick up the nearest object and place it on the counter"

### Expected Outcomes

- Robot perceives environment through vision
- Natural language commands translate to actions
- Robot completes multi-step tasks successfully

## Acceptance Criteria Met

- [X] Action generation techniques explained
- [X] Template-based and LLM-based mapping
- [X] Lab 7: Natural language task planning
- [X] Lab 8: VLA environment interaction
