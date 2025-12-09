# Action Generation from Language

## Overview

Action generation from language is the critical component that translates natural language commands into executable robot actions. This process involves mapping linguistic descriptions of tasks to specific motor commands, navigation paths, manipulation sequences, and other robot behaviors. The challenge lies in bridging the gap between abstract language and concrete physical actions while maintaining safety, efficiency, and task success.

## Instruction-to-Action Mapping Techniques

### 1. Template-Based Mapping

The simplest approach uses predefined templates that map language patterns to actions:

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
            },
            {
                'pattern': r'follow (me|the )(?P<target>\w+)',
                'action': 'follow',
                'parameters': ['target']
            }
        ]

    def map_to_action(self, command: str) -> Dict[str, Any]:
        """
        Map natural language command to robot action using templates
        """
        import re

        for template in self.action_templates:
            match = re.search(template['pattern'], command, re.IGNORECASE)
            if match:
                # Extract parameters
                params = {}
                for param in template['parameters']:
                    if param in match.groupdict():
                        params[param] = match.group(param)

                return {
                    'action': template['action'],
                    'parameters': params,
                    'confidence': 1.0,
                    'template_used': template['pattern']
                }

        # If no template matches, return a default action
        return {
            'action': 'unknown',
            'parameters': {'command': command},
            'confidence': 0.0,
            'template_used': None
        }
```

### 2. Semantic Role Labeling Approach

Using semantic role labeling to identify the roles of different entities in the command:

```python
class SemanticRoleActionMapper:
    def __init__(self):
        # Define semantic roles that are relevant for robot actions
        self.semantic_roles = {
            'A0': 'agent',  # Who performs the action (usually the robot)
            'A1': 'theme',  # What is affected by the action (object being manipulated)
            'A2': 'recipient',  # Who receives the result (destination/target)
            'LOC': 'location',  # Where the action takes place
            'DIR': 'direction',  # Direction of movement
            'MNR': 'manner',  # How the action is performed
            'TMP': 'time',  # When the action should be performed
        }

    def extract_semantic_roles(self, command: str) -> Dict[str, str]:
        """
        Extract semantic roles from command (simplified implementation)
        In practice, this would use NLP tools like AllenNLP's SRL
        """
        # This is a simplified rule-based approach
        # In practice, use a proper SRL system
        roles = {}

        # Identify common patterns
        if 'to' in command or 'at' in command:
            # Split on prepositions to identify location/theme
            parts = command.split()
            for i, part in enumerate(parts):
                if part in ['to', 'at', 'on', 'in']:
                    if i + 1 < len(parts):
                        roles['A2'] = parts[i + 1]  # recipient/location
                        if i > 0:
                            roles['A1'] = parts[i - 1]  # theme

        # Identify action verbs and their arguments
        action_verbs = {
            'bring': ['A1', 'A2'],  # bring X to Y
            'take': ['A1', 'A2'],   # take X to Y
            'put': ['A1', 'A2'],    # put X on Y
            'go': ['DIR'],          # go to X
            'get': ['A1'],          # get X
        }

        for verb, expected_roles in action_verbs.items():
            if verb in command:
                roles['action'] = verb
                break

        return roles

    def map_to_action(self, command: str) -> Dict[str, Any]:
        """
        Map command to action using semantic role labeling
        """
        roles = self.extract_semantic_roles(command)
        action_type = roles.get('action', 'unknown')

        # Map semantic roles to robot actions
        action_mapping = {
            'bring': 'fetch_and_deliver',
            'take': 'transport',
            'put': 'place',
            'go': 'navigate',
            'get': 'grasp',
            'follow': 'follow',
        }

        robot_action = action_mapping.get(action_type, 'unknown')

        # Extract parameters based on roles
        parameters = {}
        if 'A1' in roles:  # theme/object
            parameters['target_object'] = roles['A1']
        if 'A2' in roles:  # recipient/location
            parameters['destination'] = roles['A2']
        if 'DIR' in roles:  # direction
            parameters['direction'] = roles['DIR']

        return {
            'action': robot_action,
            'parameters': parameters,
            'semantic_roles': roles,
            'confidence': 0.8 if robot_action != 'unknown' else 0.3
        }
```

### 3. Neural Action Generation

Using neural networks to learn the mapping from language to actions:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class NeuralActionGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, action_space_size=20):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.language_encoder = AutoModel.from_pretrained('bert-base-uncased')

        # Action prediction head
        self.action_predictor = nn.Sequential(
            nn.Linear(self.language_encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_space_size)
        )

        # Parameter prediction head (for continuous parameters)
        self.param_predictor = nn.Sequential(
            nn.Linear(self.language_encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6)  # x, y, z, roll, pitch, yaw
        )

    def forward(self, input_ids, attention_mask):
        # Encode the language input
        encoded = self.language_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the [CLS] token representation
        cls_representation = encoded.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Predict action
        action_logits = self.action_predictor(cls_representation)

        # Predict parameters
        param_outputs = self.param_predictor(cls_representation)

        return {
            'action_logits': action_logits,
            'param_outputs': param_outputs
        }

    def generate_action(self, command: str) -> Dict[str, Any]:
        """
        Generate action from command using neural model
        """
        # Tokenize input
        inputs = self.tokenizer(
            command,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        # Forward pass
        outputs = self(**inputs)

        # Get action prediction
        action_probs = torch.softmax(outputs['action_logits'], dim=-1)
        predicted_action_idx = torch.argmax(action_probs, dim=-1).item()
        confidence = action_probs.max().item()

        # Get parameter prediction
        param_values = outputs['param_outputs'].squeeze().detach().numpy()

        # Map action index to action name (in practice, this would be a proper mapping)
        action_names = [
            'navigate', 'grasp', 'place', 'transport', 'follow',
            'inspect', 'wait', 'avoid', 'open', 'close',
            'push', 'pull', 'lift', 'lower', 'rotate',
            'wave', 'point', 'greet', 'search', 'unknown'
        ]

        predicted_action = action_names[predicted_action_idx] if predicted_action_idx < len(action_names) else 'unknown'

        return {
            'action': predicted_action,
            'parameters': {
                'position': param_values[:3].tolist(),  # x, y, z
                'orientation': param_values[3:].tolist()  # roll, pitch, yaw
            },
            'confidence': confidence,
            'action_index': predicted_action_idx
        }
```

## Grounding Language in Physical States

### 1. Visual Grounding

Connecting language to visual percepts:

```python
class VisualGroundingSystem:
    def __init__(self):
        self.object_detector = None  # Initialize with object detection model
        self.spatial_reasoner = SpatialReasoner()

    def ground_language_in_vision(self, command: str, visual_input: np.ndarray) -> Dict[str, Any]:
        """
        Ground language command in visual input
        """
        # Detect objects in the scene
        detected_objects = self.detect_objects(visual_input)

        # Parse the command to identify target objects
        target_descriptor = self.extract_object_descriptor(command)

        # Find the object that matches the descriptor
        target_object = self.find_matching_object(target_descriptor, detected_objects)

        if target_object is None:
            return {
                'success': False,
                'reason': 'Target object not found in visual scene',
                'suggested_action': 'ask_for_clarification'
            }

        # Get spatial relationships
        spatial_info = self.spatial_reasoner.get_spatial_relationships(
            target_object, detected_objects
        )

        return {
            'success': True,
            'target_object': target_object,
            'spatial_context': spatial_info,
            'grounded_command': self.update_command_with_grounding(
                command, target_object, spatial_info
            )
        }

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the image (simplified)
        """
        # This would typically use a model like YOLO, Detectron2, etc.
        # For this example, we'll simulate detection
        import random

        # Simulated object detection results
        objects = [
            {'name': 'red cup', 'bbox': [100, 100, 200, 200], 'confidence': 0.9},
            {'name': 'blue bottle', 'bbox': [300, 150, 400, 250], 'confidence': 0.85},
            {'name': 'wooden table', 'bbox': [50, 300, 450, 400], 'confidence': 0.95}
        ]

        return objects

    def extract_object_descriptor(self, command: str) -> str:
        """
        Extract object descriptor from command
        """
        # Simple keyword extraction (in practice, use more sophisticated NLP)
        import re

        # Look for color + object patterns
        color_object_pattern = r'(red|blue|green|yellow|big|small|large|wooden)\s+(\w+)'
        match = re.search(color_object_pattern, command.lower())

        if match:
            return f"{match.group(1)} {match.group(2)}"

        # Look for object names
        object_names = ['cup', 'bottle', 'table', 'chair', 'box', 'phone', 'book']
        for obj in object_names:
            if obj in command.lower():
                return obj

        return 'unknown'

    def find_matching_object(self, descriptor: str, objects: List[Dict]) -> Dict:
        """
        Find object that matches the descriptor
        """
        for obj in objects:
            if descriptor.lower() in obj['name'].lower():
                return obj

        # If exact match not found, use similarity
        best_match = None
        best_score = 0

        for obj in objects:
            score = self.calculate_descriptor_similarity(descriptor, obj['name'])
            if score > best_score:
                best_score = score
                best_match = obj

        return best_match

    def calculate_descriptor_similarity(self, desc1: str, desc2: str) -> float:
        """
        Calculate similarity between descriptors
        """
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)
```

### 2. Spatial Grounding

Understanding spatial relationships in the environment:

```python
class SpatialReasoner:
    def __init__(self):
        self.spatial_relations = ['left', 'right', 'front', 'back', 'near', 'far', 'on', 'in', 'under', 'above']

    def get_spatial_relationships(self, target_object: Dict, all_objects: List[Dict]) -> Dict[str, Any]:
        """
        Calculate spatial relationships between target object and other objects
        """
        relationships = {}

        target_center = self.get_object_center(target_object['bbox'])

        for obj in all_objects:
            if obj['name'] != target_object['name']:
                obj_center = self.get_object_center(obj['bbox'])

                relation = self.calculate_spatial_relation(target_center, obj_center)
                distance = self.calculate_distance(target_center, obj_center)

                relationships[obj['name']] = {
                    'relation': relation,
                    'distance': distance,
                    'relative_position': obj_center
                }

        return relationships

    def get_object_center(self, bbox: List[int]) -> Tuple[float, float]:
        """
        Get center coordinates of bounding box
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_spatial_relation(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> str:
        """
        Calculate spatial relation between two positions
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        # Determine horizontal relation
        if abs(dx) > abs(dy):  # More horizontal than vertical
            if dx > 0:
                return 'right'
            else:
                return 'left'
        else:  # More vertical than horizontal
            if dy > 0:
                return 'below'  # from perspective of image coordinates
            else:
                return 'above'

    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions
        """
        import math
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def ground_spatial_language(self, command: str, spatial_context: Dict) -> Dict[str, Any]:
        """
        Ground spatial language in the environment context
        """
        # Identify spatial references in command
        spatial_refs = self.extract_spatial_references(command)

        resolved_targets = {}

        for ref in spatial_refs:
            resolved = self.resolve_spatial_reference(ref, spatial_context)
            resolved_targets[ref] = resolved

        return {
            'spatial_references': spatial_refs,
            'resolved_targets': resolved_targets,
            'fully_resolved': all(resolved is not None for resolved in resolved_targets.values())
        }

    def extract_spatial_references(self, command: str) -> List[str]:
        """
        Extract spatial references from command
        """
        import re

        spatial_patterns = [
            r'the (left|right) (?:one|side|object)',
            r'the (front|back) (?:one|object)',
            r'(\w+) (?:to|on|at) the (left|right|front|back)',
            r'near (the )?(\w+)',
            r'next to (the )?(\w+)',
            r'by (the )?(\w+)'
        ]

        references = []
        for pattern in spatial_patterns:
            matches = re.findall(pattern, command, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    references.extend([m for m in match if m and m != 'the'])
                else:
                    references.append(match)

        return list(set(references))  # Remove duplicates

    def resolve_spatial_reference(self, ref: str, spatial_context: Dict) -> Dict[str, Any]:
        """
        Resolve a spatial reference to a specific object or location
        """
        # Look for exact object matches first
        for obj_name, obj_info in spatial_context.items():
            if ref.lower() in obj_name.lower():
                return {
                    'type': 'object',
                    'name': obj_name,
                    'position': obj_info['relative_position'],
                    'relation': obj_info['relation']
                }

        # If not found, check if ref is a spatial relation
        if ref.lower() in self.spatial_relations:
            # Find objects with this relation
            for obj_name, obj_info in spatial_context.items():
                if obj_info['relation'].lower() == ref.lower():
                    return {
                        'type': 'object',
                        'name': obj_name,
                        'position': obj_info['relative_position'],
                        'relation': obj_info['relation']
                    }

        return None
```

## Execution Planning Examples

### 1. Hierarchical Task Planning

```python
class HierarchicalTaskPlanner:
    def __init__(self):
        self.action_library = {
            'navigate': self.plan_navigation,
            'grasp': self.plan_grasp,
            'place': self.plan_placement,
            'transport': self.plan_transport,
            'inspect': self.plan_inspection
        }

    def plan_action_sequence(self, grounded_command: Dict) -> List[Dict[str, Any]]:
        """
        Plan a sequence of actions to execute the grounded command
        """
        action = grounded_command['action']
        parameters = grounded_command.get('parameters', {})

        if action in self.action_library:
            return self.action_library[action](parameters)
        else:
            return self.plan_generic_action(action, parameters)

    def plan_navigation(self, params: Dict) -> List[Dict[str, Any]]:
        """
        Plan navigation action sequence
        """
        destination = params.get('destination', params.get('target', 'unknown'))

        return [
            {
                'action': 'find_path_to',
                'parameters': {'target': destination},
                'description': f'Finding path to {destination}'
            },
            {
                'action': 'execute_path',
                'parameters': {'speed': 'default'},
                'description': 'Moving to destination'
            },
            {
                'action': 'localize',
                'parameters': {},
                'description': 'Confirming arrival at destination'
            }
        ]

    def plan_grasp(self, params: Dict) -> List[Dict[str, Any]]:
        """
        Plan grasping action sequence
        """
        target_object = params.get('target_object', params.get('object', 'unknown'))

        return [
            {
                'action': 'approach_object',
                'parameters': {'target': target_object},
                'description': f'Approaching {target_object}'
            },
            {
                'action': 'identify_grasp_point',
                'parameters': {'object': target_object},
                'description': 'Identifying optimal grasp point'
            },
            {
                'action': 'execute_grasp',
                'parameters': {'object': target_object, 'grasp_type': 'top_grasp'},
                'description': f'Grasping {target_object}'
            },
            {
                'action': 'verify_grasp',
                'parameters': {'object': target_object},
                'description': 'Verifying successful grasp'
            }
        ]

    def plan_placement(self, params: Dict) -> List[Dict[str, Any]]:
        """
        Plan placement action sequence
        """
        target_object = params.get('target_object', params.get('object', 'unknown'))
        surface = params.get('surface', params.get('destination', 'table'))

        return [
            {
                'action': 'navigate_to',
                'parameters': {'location': surface},
                'description': f'Navigating to {surface}'
            },
            {
                'action': 'identify_placement_point',
                'parameters': {'surface': surface},
                'description': 'Finding safe placement location'
            },
            {
                'action': 'execute_placement',
                'parameters': {'object': target_object, 'surface': surface},
                'description': f'Placing {target_object} on {surface}'
            },
            {
                'action': 'verify_placement',
                'parameters': {'object': target_object, 'surface': surface},
                'description': 'Verifying successful placement'
            }
        ]

    def plan_transport(self, params: Dict) -> List[Dict[str, Any]]:
        """
        Plan transport action sequence (pick and place)
        """
        target_object = params.get('target_object', params.get('object', 'unknown'))
        destination = params.get('destination', params.get('target', 'unknown'))

        # Combine grasp and place sequences
        grasp_sequence = self.plan_grasp({'target_object': target_object})
        navigate_sequence = [
            {
                'action': 'navigate_to',
                'parameters': {'location': destination},
                'description': f'Navigating to {destination} with {target_object}'
            }
        ]
        place_sequence = self.plan_placement({'target_object': target_object, 'surface': destination})

        return grasp_sequence + navigate_sequence + place_sequence

    def plan_inspection(self, params: Dict) -> List[Dict[str, Any]]:
        """
        Plan inspection action sequence
        """
        target = params.get('target', 'unknown')

        return [
            {
                'action': 'navigate_to',
                'parameters': {'location': target},
                'description': f'Navigating to {target} for inspection'
            },
            {
                'action': 'capture_images',
                'parameters': {'view_points': ['front', 'left', 'right']},
                'description': 'Capturing images from multiple angles'
            },
            {
                'action': 'analyze_object',
                'parameters': {'target': target},
                'description': 'Analyzing captured images'
            },
            {
                'action': 'report_findings',
                'parameters': {'target': target},
                'description': 'Reporting inspection results'
            }
        ]

    def plan_generic_action(self, action: str, params: Dict) -> List[Dict[str, Any]]:
        """
        Plan for unknown actions (fallback)
        """
        return [
            {
                'action': 'unknown_action',
                'parameters': {'action_name': action, 'params': params},
                'description': f'Unknown action: {action}',
                'status': 'requires_human_intervention'
            }
        ]
```

### 2. Execution Monitoring and Adaptation

```python
class ExecutionMonitor:
    def __init__(self):
        self.action_history = []
        self.failure_count = 0
        self.max_failures = 3

    def execute_action_sequence(self, action_sequence: List[Dict], robot_interface) -> Dict[str, Any]:
        """
        Execute a sequence of actions with monitoring and adaptation
        """
        results = []

        for i, action in enumerate(action_sequence):
            try:
                # Execute the action
                result = self.execute_single_action(action, robot_interface)

                # Monitor execution
                monitoring_result = self.monitor_action(action, result)

                # Log the result
                action_result = {
                    'action': action,
                    'result': result,
                    'monitoring': monitoring_result,
                    'timestamp': time.time()
                }

                results.append(action_result)

                # Check for failure
                if not monitoring_result.get('success', True):
                    self.failure_count += 1

                    if self.failure_count >= self.max_failures:
                        return {
                            'success': False,
                            'results': results,
                            'failure_reason': 'Too many failures',
                            'final_status': 'aborted'
                        }

                    # Try to recover
                    recovery_result = self.attempt_recovery(action, result, robot_interface)
                    if recovery_result['success']:
                        results.append({
                            'action': {'action': 'recovery', 'type': recovery_result['recovery_type']},
                            'result': recovery_result,
                            'timestamp': time.time()
                        })
                    else:
                        return {
                            'success': False,
                            'results': results,
                            'failure_reason': 'Recovery failed',
                            'final_status': 'failed'
                        }
                else:
                    self.failure_count = 0  # Reset on success

            except Exception as e:
                # Handle unexpected errors
                error_result = {
                    'action': action,
                    'result': {'success': False, 'error': str(e)},
                    'monitoring': {'success': False, 'error': str(e)},
                    'timestamp': time.time()
                }
                results.append(error_result)

                return {
                    'success': False,
                    'results': results,
                    'failure_reason': f'Unexpected error: {str(e)}',
                    'final_status': 'error'
                }

        return {
            'success': True,
            'results': results,
            'failure_reason': None,
            'final_status': 'completed'
        }

    def execute_single_action(self, action: Dict, robot_interface) -> Dict[str, Any]:
        """
        Execute a single action using the robot interface
        """
        action_name = action['action']
        parameters = action.get('parameters', {})

        # Map action name to robot interface method
        action_methods = {
            'navigate_to': robot_interface.navigate_to,
            'approach_object': robot_interface.approach_object,
            'execute_grasp': robot_interface.execute_grasp,
            'execute_placement': robot_interface.execute_placement,
            'find_path_to': robot_interface.find_path,
            'execute_path': robot_interface.execute_path,
            'localize': robot_interface.localize,
            'identify_grasp_point': robot_interface.identify_grasp_point,
            'verify_grasp': robot_interface.verify_grasp,
            'verify_placement': robot_interface.verify_placement
        }

        if action_name in action_methods:
            try:
                result = action_methods[action_name](**parameters)
                return {
                    'success': True,
                    'result': result,
                    'action_executed': action_name
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'action_executed': action_name
                }
        else:
            return {
                'success': False,
                'error': f'Unknown action: {action_name}',
                'action_executed': action_name
            }

    def monitor_action(self, action: Dict, result: Dict) -> Dict[str, Any]:
        """
        Monitor action execution and verify success
        """
        action_name = action['action']

        # Basic success check
        success = result.get('success', False)

        # Additional verification based on action type
        verification = {}

        if action_name in ['execute_grasp', 'verify_grasp']:
            # Check if object is actually grasped
            verification['object_grasped'] = result.get('object_grasped', False)
            success = success and verification['object_grasped']

        elif action_name in ['execute_placement', 'verify_placement']:
            # Check if object is properly placed
            verification['object_placed'] = result.get('object_placed', False)
            success = success and verification['object_placed']

        elif action_name == 'navigate_to':
            # Check if close enough to target
            distance_to_target = result.get('distance_to_target', float('inf'))
            verification['close_to_target'] = distance_to_target < 0.1  # within 10cm
            success = success and verification['close_to_target']

        return {
            'success': success,
            'verification': verification,
            'result_details': result
        }

    def attempt_recovery(self, failed_action: Dict, failure_result: Dict, robot_interface) -> Dict[str, Any]:
        """
        Attempt to recover from action failure
        """
        action_name = failed_action['action']

        # Different recovery strategies based on action type
        if action_name == 'execute_grasp':
            # Try a different grasp approach
            return self.recovery_grasp(failed_action, failure_result, robot_interface)
        elif action_name == 'navigate_to':
            # Try alternative path or approach
            return self.recovery_navigation(failed_action, failure_result, robot_interface)
        elif action_name == 'execute_placement':
            # Try different placement location
            return self.recovery_placement(failed_action, failure_result, robot_interface)
        else:
            # For other actions, try again with slight variations
            return self.recovery_generic(failed_action, failure_result, robot_interface)

    def recovery_grasp(self, action: Dict, result: Dict, robot_interface) -> Dict[str, Any]:
        """
        Recovery strategy for grasp failures
        """
        original_params = action.get('parameters', {})
        target_object = original_params.get('object')

        # Try a different grasp type
        alternative_grasp_types = ['side_grasp', 'top_grasp', 'pinch_grasp']
        current_grasp = original_params.get('grasp_type', 'top_grasp')

        for grasp_type in alternative_grasp_types:
            if grasp_type != current_grasp:
                try:
                    recovery_result = robot_interface.execute_grasp(
                        object=target_object,
                        grasp_type=grasp_type
                    )

                    if recovery_result.get('success', False):
                        return {
                            'success': True,
                            'recovery_type': 'alternative_grasp',
                            'grasp_type_used': grasp_type,
                            'result': recovery_result
                        }
                except:
                    continue

        return {
            'success': False,
            'recovery_type': 'grasp_recovery',
            'attempts': alternative_grasp_types
        }

    def recovery_navigation(self, action: Dict, result: Dict, robot_interface) -> Dict[str, Any]:
        """
        Recovery strategy for navigation failures
        """
        original_params = action.get('parameters', {})
        target = original_params.get('location') or original_params.get('target')

        try:
            # Try to find an alternative path
            alternative_path = robot_interface.find_alternative_path(target)
            if alternative_path:
                nav_result = robot_interface.execute_path(alternative_path)
                if nav_result.get('success', False):
                    return {
                        'success': True,
                        'recovery_type': 'alternative_path',
                        'result': nav_result
                    }
        except:
            pass

        return {
            'success': False,
            'recovery_type': 'navigation_recovery'
        }

    def recovery_generic(self, action: Dict, result: Dict, robot_interface) -> Dict[str, Any]:
        """
        Generic recovery strategy
        """
        # Simply try the action again
        try:
            retry_result = self.execute_single_action(action, robot_interface)
            if retry_result.get('success', False):
                return {
                    'success': True,
                    'recovery_type': 'retry',
                    'result': retry_result
                }
        except:
            pass

        return {
            'success': False,
            'recovery_type': 'generic_recovery'
        }
```

## Integration with LLM-Based Systems

### 1. LLM-Enhanced Action Generation

```python
class LLMEnhancedActionGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.action_planner = HierarchicalTaskPlanner()
        self.execution_monitor = ExecutionMonitor()

    def generate_and_execute(self, command: str, robot_context: Dict) -> Dict[str, Any]:
        """
        Generate and execute actions using LLM assistance
        """
        # Use LLM to interpret the command and generate plan
        plan = self.llm_generate_plan(command, robot_context)

        # Execute the plan
        execution_result = self.execution_monitor.execute_action_sequence(
            plan, robot_context.get('robot_interface')
        )

        return {
            'command': command,
            'generated_plan': plan,
            'execution_result': execution_result,
            'overall_success': execution_result['success']
        }

    def llm_generate_plan(self, command: str, context: Dict) -> List[Dict[str, Any]]:
        """
        Use LLM to generate an action plan for the command
        """
        prompt = f"""
        You are a helpful assistant that converts natural language commands into robot action plans.
        Given the following command and context, generate a detailed action plan.

        Command: "{command}"

        Robot Capabilities: {context.get('capabilities', 'unknown')}
        Current State: {context.get('state', 'unknown')}
        Environment: {context.get('environment', 'unknown')}
        Available Objects: {context.get('objects', 'unknown')}

        Please respond with a list of actions in JSON format:
        [
            {{
                "action": "action_name",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "description": "Brief description of what this action does"
            }},
            ...
        ]

        Each action should be a discrete, executable step. Be specific about targets and parameters.
        """

        try:
            response = self.llm_client.generate(prompt, max_tokens=1000, temperature=0.3)

            # Parse the response
            import json
            plan = json.loads(response)

            # Validate the plan format
            validated_plan = self.validate_action_plan(plan)

            return validated_plan

        except Exception as e:
            print(f"LLM plan generation failed: {e}")
            # Fallback to traditional planning
            return self.fallback_plan(command, context)

    def validate_action_plan(self, plan: List[Dict]) -> List[Dict[str, Any]]:
        """
        Validate and normalize the action plan
        """
        validated_plan = []

        for action in plan:
            if isinstance(action, dict) and 'action' in action:
                validated_action = {
                    'action': action['action'],
                    'parameters': action.get('parameters', {}),
                    'description': action.get('description', '')
                }
                validated_plan.append(validated_action)

        return validated_plan

    def fallback_plan(self, command: str, context: Dict) -> List[Dict[str, Any]]:
        """
        Fallback to traditional planning if LLM fails
        """
        # Use the semantic parser and traditional planner
        nlu = RoboticNLU()  # Assuming we have this from previous section
        parsed = nlu.process_command(command, context)

        if parsed['ready_for_execution']:
            return self.action_planner.plan_action_sequence(parsed['parsed'])
        else:
            # If parsing fails, return a simple "ask for clarification" plan
            return [{
                'action': 'request_clarification',
                'parameters': {'original_command': command},
                'description': 'Request clarification from user'
            }]
```

### 2. Safety and Validation Layer

```python
class SafeActionGenerator:
    def __init__(self):
        self.safety_rules = self.load_safety_rules()
        self.action_validator = ActionValidator()

    def generate_safe_actions(self, command: str, context: Dict) -> Dict[str, Any]:
        """
        Generate actions with safety validation
        """
        # Generate initial action plan
        generator = LLMEnhancedActionGenerator(context.get('llm_client'))
        plan = generator.llm_generate_plan(command, context)

        # Validate for safety
        validation_result = self.validate_safety(plan, context)

        if validation_result['safe']:
            return {
                'success': True,
                'action_plan': plan,
                'safety_validation': validation_result,
                'ready_to_execute': True
            }
        else:
            return {
                'success': False,
                'action_plan': plan,
                'safety_validation': validation_result,
                'ready_to_execute': False,
                'suggested_revisions': validation_result.get('suggested_revisions', [])
            }

    def validate_safety(self, action_plan: List[Dict], context: Dict) -> Dict[str, Any]:
        """
        Validate action plan for safety
        """
        issues = []
        suggestions = []

        for i, action in enumerate(action_plan):
            action_issues = self.check_action_safety(action, context)
            if action_issues:
                issues.extend([f"Action {i}: {issue}" for issue in action_issues])

                # Generate suggestions for safer alternatives
                suggestion = self.generate_safety_suggestion(action, action_issues)
                if suggestion:
                    suggestions.append(suggestion)

        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'suggested_revisions': suggestions
        }

    def check_action_safety(self, action: Dict, context: Dict) -> List[str]:
        """
        Check if an action is safe to execute
        """
        issues = []
        action_name = action['action']
        params = action.get('parameters', {})

        # Check for collision risk
        if action_name in ['navigate', 'move_to', 'approach']:
            target_location = params.get('target', params.get('location'))
            if self.would_cause_collision(action_name, target_location, context):
                issues.append(f"Collision risk when moving to {target_location}")

        # Check for unsafe manipulation
        if action_name in ['grasp', 'manipulate']:
            target_object = params.get('target', params.get('object'))
            if self.is_unsafe_to_manipulate(target_object, context):
                issues.append(f"Unsafe to manipulate object: {target_object}")

        # Check for safety rule violations
        for rule in self.safety_rules:
            if self.violates_safety_rule(action, rule, context):
                issues.append(f"Violates safety rule: {rule['description']}")

        return issues

    def load_safety_rules(self) -> List[Dict]:
        """
        Load predefined safety rules
        """
        return [
            {
                'name': 'avoid_people',
                'description': 'Never navigate close to or manipulate around people',
                'priority': 'high',
                'condition': lambda action, context: self.check_people_proximity(action, context)
            },
            {
                'name': 'respect_boundaries',
                'description': 'Stay within designated operational boundaries',
                'priority': 'high',
                'condition': lambda action, context: self.check_boundary_violation(action, context)
            },
            {
                'name': 'handle_fragile_objects_carefully',
                'description': 'Use gentle manipulation for fragile objects',
                'priority': 'medium',
                'condition': lambda action, context: self.check_fragile_object_handling(action, context)
            }
        ]

    def would_cause_collision(self, action_name: str, target: str, context: Dict) -> bool:
        """
        Check if action would cause collision
        """
        # Implementation would check robot's map, path planning, etc.
        return False  # Simplified for example

    def is_unsafe_to_manipulate(self, target_object: str, context: Dict) -> bool:
        """
        Check if object is unsafe to manipulate
        """
        # Check if object is hot, sharp, toxic, etc.
        dangerous_objects = context.get('dangerous_objects', [])
        return target_object in dangerous_objects

    def violates_safety_rule(self, action: Dict, rule: Dict, context: Dict) -> bool:
        """
        Check if action violates a specific safety rule
        """
        return rule['condition'](action, context)

    def generate_safety_suggestion(self, action: Dict, issues: List[str]) -> Dict[str, Any]:
        """
        Generate safety suggestion for problematic action
        """
        # For now, return a generic suggestion
        # In practice, this would have specific suggestions based on the issues
        return {
            'original_action': action,
            'safety_issues': issues,
            'suggested_alternative': 'request_human_approval',
            'description': 'Action flagged for safety review'
        }
```

## Performance Evaluation

### 1. Action Generation Metrics

```python
class ActionGenerationEvaluator:
    def __init__(self):
        self.metrics = {
            'success_rate': [],
            'execution_time': [],
            'plan_quality': [],
            'safety_violations': [],
            'human_interventions': []
        }

    def evaluate_action_generation(self, command: str, expected_actions: List[Dict],
                                 generated_actions: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the quality of generated actions
        """
        evaluation = {}

        # Success rate (whether the action achieves the goal)
        evaluation['success'] = self.evaluate_success(command, expected_actions, generated_actions)

        # Plan quality (similarity to expected plan)
        evaluation['plan_similarity'] = self.calculate_plan_similarity(
            expected_actions, generated_actions
        )

        # Safety evaluation
        evaluation['safety_score'] = self.evaluate_safety(generated_actions)

        # Efficiency (number of actions needed)
        evaluation['efficiency'] = len(expected_actions) / len(generated_actions) if generated_actions else 0

        # Update metrics
        for key, value in evaluation.items():
            if key in self.metrics:
                self.metrics[key].append(value)

        return evaluation

    def evaluate_success(self, command: str, expected: List[Dict], generated: List[Dict]) -> bool:
        """
        Evaluate if the generated actions would successfully complete the command
        """
        # This would typically involve simulation or execution
        # For this example, we'll use a simple comparison
        if len(expected) == 0 and len(generated) == 0:
            return True
        elif len(expected) == 0 or len(generated) == 0:
            return False

        # Compare action types (simplified)
        expected_actions = [a['action'] for a in expected]
        generated_actions = [a['action'] for a in generated]

        # Check if all required actions are present
        required_actions = set(expected_actions)
        provided_actions = set(generated_actions)

        return required_actions.issubset(provided_actions)

    def calculate_plan_similarity(self, expected: List[Dict], generated: List[Dict]) -> float:
        """
        Calculate similarity between expected and generated plans
        """
        if not expected and not generated:
            return 1.0
        if not expected or not generated:
            return 0.0

        # Use sequence alignment to compare plans
        import difflib
        expected_seq = [f"{a['action']}_{list(a.get('parameters', {}).keys())}" for a in expected]
        generated_seq = [f"{a['action']}_{list(a.get('parameters', {}).keys())}" for a in generated]

        similarity = difflib.SequenceMatcher(None, expected_seq, generated_seq).ratio()
        return similarity

    def evaluate_safety(self, actions: List[Dict]) -> float:
        """
        Evaluate safety of action sequence
        """
        safe_action_types = {'navigate', 'grasp', 'place', 'wait', 'stop'}
        unsafe_action_types = {'jump', 'run_fast', 'grab_forcefully'}

        safe_count = sum(1 for a in actions if a['action'] in safe_action_types)
        unsafe_count = sum(1 for a in actions if a['action'] in unsafe_action_types)

        total_actions = len(actions)
        if total_actions == 0:
            return 1.0

        safety_score = safe_count / total_actions
        if unsafe_count > 0:
            safety_score *= 0.5  # Penalize for unsafe actions

        return max(0.0, safety_score)
```

## Acceptance Criteria Met

- [X] Instruction-to-action mapping techniques
- [X] Grounding language in physical states
- [X] Execution planning examples