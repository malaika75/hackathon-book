# Natural Language Understanding

## Overview

Natural Language Understanding (NLU) is a critical component in robotics that enables robots to interpret and make sense of human language commands. In the context of Visual Language-Action (VLA) models, NLU bridges the gap between human communication and robot execution, allowing for intuitive human-robot interaction. This section covers the techniques, challenges, and implementations of NLU for robotic command interpretation.

## Core NLU Components for Robotics

### 1. Intent Recognition

Intent recognition identifies the underlying purpose or goal of a human command. In robotics, intents are typically action-oriented:

```python
class IntentRecognizer:
    def __init__(self):
        self.intent_map = {
            'navigation': ['go to', 'move to', 'navigate to', 'walk to', 'drive to'],
            'manipulation': ['pick up', 'grasp', 'take', 'get', 'lift', 'put down', 'place', 'set'],
            'interaction': ['greet', 'introduce', 'meet', 'talk to', 'say hello'],
            'information': ['what is', 'where is', 'find', 'locate', 'show me', 'describe'],
            'social': ['follow me', 'wait', 'stop', 'come here', 'wait for me']
        }

    def recognize_intent(self, utterance: str) -> Dict[str, Any]:
        """
        Recognize the intent from a natural language utterance
        """
        utterance_lower = utterance.lower()

        best_intent = None
        best_score = 0

        for intent, keywords in self.intent_map.items():
            score = self.calculate_intent_score(utterance_lower, keywords)
            if score > best_score:
                best_score = score
                best_intent = intent

        return {
            'intent': best_intent,
            'confidence': best_score,
            'original_utterance': utterance
        }

    def calculate_intent_score(self, utterance: str, keywords: List[str]) -> float:
        """
        Calculate how well the utterance matches the intent keywords
        """
        score = 0
        for keyword in keywords:
            if keyword in utterance:
                score += 1

        return score / len(keywords)
```

### 2. Entity Extraction

Entity extraction identifies specific objects, locations, or parameters mentioned in commands:

```python
class EntityExtractor:
    def __init__(self):
        self.object_categories = [
            'cup', 'bottle', 'book', 'phone', 'keys', 'plate', 'fork', 'spoon',
            'chair', 'table', 'door', 'window', 'person', 'robot', 'box'
        ]
        self.location_keywords = [
            'kitchen', 'living room', 'bedroom', 'bathroom', 'office',
            'corridor', 'entrance', 'exit', 'left', 'right', 'front', 'back'
        ]

    def extract_entities(self, utterance: str, context: Dict = None) -> Dict[str, List[str]]:
        """
        Extract entities from the utterance
        """
        entities = {
            'objects': [],
            'locations': [],
            'people': [],
            'quantities': [],
            'descriptors': []
        }

        words = utterance.lower().split()

        # Extract objects
        for obj in self.object_categories:
            if obj in utterance.lower():
                entities['objects'].append(obj)

        # Extract locations
        for loc in self.location_keywords:
            if loc in utterance.lower():
                entities['locations'].append(loc)

        # Extract people (simple heuristic)
        if 'person' in utterance.lower() or 'someone' in utterance.lower():
            entities['people'].append('person')

        # Extract quantities (numbers)
        import re
        numbers = re.findall(r'\d+', utterance)
        entities['quantities'] = [int(n) for n in numbers]

        # Extract descriptors (adjectives)
        descriptors = self.extract_descriptors(words)
        entities['descriptors'] = descriptors

        return entities

    def extract_descriptors(self, words: List[str]) -> List[str]:
        """
        Extract potential descriptors (simple approach)
        """
        common_adjectives = ['red', 'blue', 'green', 'large', 'small', 'big', 'little',
                           'left', 'right', 'front', 'back', 'near', 'far', 'top', 'bottom',
                           'middle', 'center', 'old', 'new', 'clean', 'dirty']

        return [word for word in words if word in common_adjectives]
```

### 3. Semantic Parsing

Semantic parsing converts natural language into structured representations that robots can understand:

```python
class SemanticParser:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.entity_extractor = EntityExtractor()

    def parse_command(self, utterance: str, context: Dict = None) -> Dict[str, Any]:
        """
        Parse natural language command into structured format
        """
        # Recognize intent
        intent_result = self.intent_recognizer.recognize_intent(utterance)

        # Extract entities
        entities = self.entity_extractor.extract_entities(utterance, context)

        # Create structured representation
        structured_command = {
            'intent': intent_result['intent'],
            'entities': entities,
            'confidence': intent_result['confidence'],
            'original_utterance': utterance,
            'parsed_at': time.time()
        }

        # Add spatial relationships if present
        structured_command['spatial_relationships'] = self.extract_spatial_relationships(utterance)

        # Add temporal information
        structured_command['temporal_info'] = self.extract_temporal_info(utterance)

        return structured_command

    def extract_spatial_relationships(self, utterance: str) -> List[Dict[str, str]]:
        """
        Extract spatial relationships like 'left of', 'next to', 'behind'
        """
        spatial_patterns = [
            (r'left of (\w+)', 'left_of'),
            (r'right of (\w+)', 'right_of'),
            (r'next to (\w+)', 'next_to'),
            (r'behind (\w+)', 'behind'),
            (r'in front of (\w+)', 'in_front_of'),
            (r'between (\w+) and (\w+)', 'between')
        ]

        relationships = []
        utterance_lower = utterance.lower()

        for pattern, rel_type in spatial_patterns:
            import re
            matches = re.findall(pattern, utterance_lower)
            for match in matches:
                if isinstance(match, tuple):
                    relationships.append({
                        'type': rel_type,
                        'arguments': list(match)
                    })
                else:
                    relationships.append({
                        'type': rel_type,
                        'arguments': [match]
                    })

        return relationships

    def extract_temporal_info(self, utterance: str) -> Dict[str, Any]:
        """
        Extract temporal information like 'now', 'later', 'after'
        """
        temporal_indicators = {
            'immediate': ['now', 'immediately', 'right now', 'at once'],
            'delayed': ['later', 'after', 'when', 'until'],
            'frequency': ['always', 'sometimes', 'often', 'never']
        }

        temporal_info = {}
        utterance_lower = utterance.lower()

        for category, indicators in temporal_indicators.items():
            for indicator in indicators:
                if indicator in utterance_lower:
                    if category not in temporal_info:
                        temporal_info[category] = []
                    temporal_info[category].append(indicator)

        return temporal_info
```

## Command Parsing and Interpretation

### 1. Grammar-Based Parsing

Using formal grammars to parse robotic commands:

```python
import pyparsing as pp

class GrammarBasedParser:
    def __init__(self):
        self.setup_grammar()

    def setup_grammar(self):
        """
        Define grammar for robotic commands
        """
        # Define basic elements
        verb = pp.oneOf("go pick place navigate move grasp take put follow wait")
        article = pp.oneOf("a an the")
        adjective = pp.oneOf("red blue green big small left right")
        noun = pp.oneOf("cup bottle book phone keys plate fork spoon chair table door person robot")
        location = pp.oneOf("kitchen living_room bedroom bathroom office")
        direction = pp.oneOf("left right front back")
        spatial_rel = pp.oneOf("of to near by")

        # Build command structure
        object_spec = pp.Optional(article) + pp.Optional(adjective) + noun
        location_spec = location | (direction + spatial_rel + noun)

        # Main command patterns
        self.move_command = verb + location_spec
        self.grasp_command = verb + object_spec
        self.complex_command = verb + object_spec + pp.Optional(pp.oneOf("at in on")) + location_spec

        # Combine all patterns
        self.command_grammar = self.move_command | self.grasp_command | self.complex_command

    def parse(self, command: str) -> Dict[str, Any]:
        """
        Parse command using defined grammar
        """
        try:
            result = self.command_grammar.parseString(command)
            return {
                'success': True,
                'parsed': result.asList(),
                'command_type': self.classify_command(result)
            }
        except pp.ParseException as e:
            return {
                'success': False,
                'error': str(e),
                'original_command': command
            }

    def classify_command(self, parsed_result: pp.ParseResults) -> str:
        """
        Classify the type of command based on parse results
        """
        if len(parsed_result) >= 1:
            verb = parsed_result[0].lower()
            if verb in ['go', 'move', 'navigate']:
                return 'navigation'
            elif verb in ['pick', 'grasp', 'take', 'get']:
                return 'manipulation'
            elif verb in ['place', 'put', 'set']:
                return 'placement'
            elif verb in ['follow', 'wait']:
                return 'social'
        return 'unknown'
```

### 2. Neural Semantic Parsing

Using neural networks for more flexible parsing:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class NeuralSemanticParser(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_intent_classes=5):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Intent classification head
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intent_classes)

        # Named entity recognition head
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, 10)  # 10 NER tags

        # Spatial relation classifier
        self.spatial_classifier = nn.Linear(self.bert.config.hidden_size, 6)  # 6 spatial relations

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Sequence output for NER
        sequence_output = outputs.last_hidden_state

        # Pooler output for classification
        pooled_output = outputs.pooler_output

        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)

        # NER tagging
        ner_logits = self.ner_classifier(sequence_output)

        # Spatial relation (simplified - in practice would need more complex structure)
        spatial_logits = self.spatial_classifier(pooled_output)

        return {
            'intent_logits': intent_logits,
            'ner_logits': ner_logits,
            'spatial_logits': spatial_logits
        }

    def parse_command(self, command: str) -> Dict[str, Any]:
        """
        Parse command using neural model
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

        # Convert logits to predictions
        intent_pred = torch.argmax(outputs['intent_logits'], dim=-1).item()
        ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)

        # Map to labels (simplified)
        intent_labels = ['navigation', 'manipulation', 'social', 'information', 'other']
        ner_labels = ['O', 'B-OBJ', 'I-OBJ', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-DES', 'I-DES', 'PAD']

        return {
            'intent': intent_labels[intent_pred],
            'entities': self.extract_entities_from_ner(command, ner_preds[0], ner_labels),
            'confidence': torch.softmax(outputs['intent_logits'], dim=-1).max().item()
        }

    def extract_entities_from_ner(self, command: str, ner_tags, ner_labels: List[str]) -> Dict[str, List[str]]:
        """
        Extract entities based on NER tags
        """
        tokens = self.tokenizer.tokenize(command)
        entities = {'objects': [], 'locations': [], 'people': [], 'descriptors': []}

        current_entity = []
        current_type = None

        for token, tag_idx in zip(tokens, ner_tags):
            tag = ner_labels[tag_idx.item()]

            if tag.startswith('B-'):  # Beginning of entity
                if current_entity:  # Save previous entity
                    entity_text = ' '.join(current_entity)
                    entity_type = current_type.split('-')[1].lower()
                    if entity_type in entities:
                        entities[entity_type].append(entity_text)

                current_entity = [token]
                current_type = tag[2:]  # Remove 'B-' prefix
            elif tag.startswith('I-') and current_type and tag[2:] == current_type[2:]:  # Inside same entity
                current_entity.append(token)
            else:  # Outside entity
                if current_entity:  # Save current entity
                    entity_text = ' '.join(current_entity)
                    entity_type = current_type.split('-')[1].lower()
                    if entity_type in entities:
                        entities[entity_type].append(entity_text)
                    current_entity = []
                    current_type = None

        # Save last entity if exists
        if current_entity:
            entity_text = ' '.join(current_entity)
            entity_type = current_type.split('-')[1].lower()
            if entity_type in entities:
                entities[entity_type].append(entity_text)

        return entities
```

## Context-Aware Understanding

### 1. Context Integration

Incorporating environmental and situational context:

```python
class ContextAwareNLU:
    def __init__(self):
        self.semantic_parser = SemanticParser()
        self.context_buffer = {}

    def interpret_command_with_context(self, command: str, robot_context: Dict) -> Dict[str, Any]:
        """
        Interpret command using robot's current context
        """
        # Parse the command
        parsed_command = self.semantic_parser.parse_command(command, robot_context)

        # Enhance with context
        enhanced_command = self.enhance_with_context(parsed_command, robot_context)

        # Resolve ambiguities using context
        resolved_command = self.resolve_ambiguities(enhanced_command, robot_context)

        return resolved_command

    def enhance_with_context(self, parsed_command: Dict, robot_context: Dict) -> Dict:
        """
        Enhance parsed command with contextual information
        """
        enhanced = parsed_command.copy()

        # Add current location context
        if 'location' in robot_context:
            enhanced['current_location'] = robot_context['location']

        # Add visible objects context
        if 'visible_objects' in robot_context:
            enhanced['visible_objects'] = robot_context['visible_objects']

        # Add robot capabilities context
        if 'capabilities' in robot_context:
            enhanced['robot_capabilities'] = robot_context['capabilities']

        # Add temporal context
        enhanced['timestamp'] = time.time()

        return enhanced

    def resolve_ambiguities(self, command: Dict, context: Dict) -> Dict:
        """
        Resolve ambiguous references using context
        """
        resolved = command.copy()

        # Resolve pronouns
        resolved['entities'] = self.resolve_pronouns(
            resolved['entities'], context
        )

        # Disambiguate objects
        resolved['entities']['objects'] = self.disambiguate_objects(
            resolved['entities']['objects'], context
        )

        # Disambiguate locations
        resolved['entities']['locations'] = self.disambiguate_locations(
            resolved['entities']['locations'], context
        )

        return resolved

    def resolve_pronouns(self, entities: Dict, context: Dict) -> Dict:
        """
        Resolve pronouns like 'it', 'them', 'there' using context
        """
        # This is a simplified example
        # In practice, this would use more sophisticated coreference resolution
        if 'it' in entities.get('objects', []):
            # Find the most recently mentioned or most salient object
            if context.get('last_seen_object'):
                entities['objects'].remove('it')
                entities['objects'].append(context['last_seen_object'])

        return entities

    def disambiguate_objects(self, objects: List[str], context: Dict) -> List[str]:
        """
        Disambiguate object references using context
        """
        if not context.get('visible_objects'):
            return objects

        visible_objects = context['visible_objects']
        disambiguated = []

        for obj in objects:
            # Find best match among visible objects
            best_match = self.find_best_object_match(obj, visible_objects)
            if best_match:
                disambiguated.append(best_match)
            else:
                disambiguated.append(obj)  # Keep original if no match found

        return disambiguated

    def find_best_object_match(self, query: str, candidates: List[Dict]) -> str:
        """
        Find the best matching object from candidates
        """
        best_score = 0
        best_match = None

        for candidate in candidates:
            score = self.calculate_object_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate.get('name', candidate.get('id', 'unknown'))

        return best_match

    def calculate_object_similarity(self, query: str, candidate: Dict) -> float:
        """
        Calculate similarity between query and candidate object
        """
        candidate_name = candidate.get('name', '').lower()
        candidate_type = candidate.get('type', '').lower()
        candidate_attributes = candidate.get('attributes', [])

        score = 0

        # Exact match
        if query == candidate_name or query == candidate_type:
            return 1.0

        # Partial matches
        if query in candidate_name or candidate_name in query:
            score += 0.6

        if query in candidate_type or candidate_type in query:
            score += 0.4

        # Attribute matches
        for attr in candidate_attributes:
            if query == attr.lower():
                score += 0.3

        return min(score, 1.0)
```

### 2. Multi-turn Understanding

Handling commands that span multiple interactions:

```python
class MultiTurnNLU:
    def __init__(self, max_context_length=10):
        self.context_history = []
        self.max_context_length = max_context_length
        self.nlu_system = ContextAwareNLU()

    def process_utterance(self, utterance: str, robot_context: Dict) -> Dict[str, Any]:
        """
        Process an utterance in multi-turn context
        """
        # Add current utterance to history
        self.context_history.append({
            'utterance': utterance,
            'timestamp': time.time(),
            'turn_type': 'user'
        })

        # Keep only recent history
        if len(self.context_history) > self.max_context_length:
            self.context_history = self.context_history[-self.max_context_length:]

        # Create conversation context
        conversation_context = self.build_conversation_context()

        # Interpret with full context
        interpretation = self.nlu_system.interpret_command_with_context(
            utterance,
            {**robot_context, **conversation_context}
        )

        # Store interpretation in context
        self.context_history[-1]['interpretation'] = interpretation

        return interpretation

    def build_conversation_context(self) -> Dict:
        """
        Build context from conversation history
        """
        context = {
            'conversation_history': self.context_history[-5:],  # Last 5 turns
            'previous_utterances': [turn['utterance'] for turn in self.context_history[-3:]],
            'last_intent': self.get_last_intent(),
            'last_entities': self.get_last_entities(),
            'conversation_topic': self.identify_conversation_topic()
        }

        return context

    def get_last_intent(self) -> str:
        """
        Get the intent from the last interpreted utterance
        """
        for turn in reversed(self.context_history):
            if 'interpretation' in turn:
                return turn['interpretation'].get('intent', 'unknown')
        return 'unknown'

    def get_last_entities(self) -> Dict:
        """
        Get entities from the last interpreted utterance
        """
        for turn in reversed(self.context_history):
            if 'interpretation' in turn:
                return turn['interpretation'].get('entities', {})
        return {}

    def identify_conversation_topic(self) -> str:
        """
        Identify the current conversation topic
        """
        # Simple topic identification based on recent intents
        recent_intents = []
        for turn in self.context_history[-5:]:
            if 'interpretation' in turn:
                intent = turn['interpretation'].get('intent')
                if intent:
                    recent_intents.append(intent)

        if not recent_intents:
            return 'general'

        # Most common intent in recent turns
        from collections import Counter
        intent_counts = Counter(recent_intents)
        return intent_counts.most_common(1)[0][0]
```

## Error Handling and Clarification

### 1. Ambiguity Detection

Detecting when commands are ambiguous and need clarification:

```python
class AmbiguityDetector:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.ambiguity_patterns = [
            r'it',  # Pronouns
            r'there',  # Vague locations
            r'something',  # Vague objects
            r'over there',  # Vague spatial references
            r'that one',  # Vague references
        ]

    def detect_ambiguity(self, parsed_command: Dict) -> Dict[str, Any]:
        """
        Detect ambiguities in the parsed command
        """
        ambiguities = {
            'low_confidence': parsed_command.get('confidence', 1.0) < self.confidence_threshold,
            'vague_references': self.find_vague_references(parsed_command),
            'missing_information': self.find_missing_info(parsed_command),
            'multiple_interpretations': self.find_multiple_interpretations(parsed_command)
        }

        # Overall ambiguity score
        ambiguity_score = self.calculate_ambiguity_score(ambiguities)
        ambiguities['overall_score'] = ambiguity_score
        ambiguities['needs_clarification'] = ambiguity_score > 0.5

        return ambiguities

    def find_vague_references(self, parsed_command: Dict) -> List[str]:
        """
        Find vague references in the command
        """
        vague_refs = []
        utterance = parsed_command.get('original_utterance', '').lower()

        import re
        for pattern in self.ambiguity_patterns:
            matches = re.findall(pattern, utterance)
            vague_refs.extend(matches)

        # Check for ambiguous entities
        entities = parsed_command.get('entities', {})
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity in ['it', 'that', 'there', 'something', 'thing']:
                    vague_refs.append(entity)

        return vague_refs

    def find_missing_info(self, parsed_command: Dict) -> List[str]:
        """
        Find missing information required for execution
        """
        missing_info = []
        intent = parsed_command.get('intent')

        if intent == 'navigation' and not parsed_command.get('entities', {}).get('locations'):
            missing_info.append('destination')

        if intent == 'manipulation' and not parsed_command.get('entities', {}).get('objects'):
            missing_info.append('target_object')

        # Check for spatial relationships
        if 'where' in parsed_command.get('original_utterance', '').lower():
            missing_info.append('spatial_reference')

        return missing_info

    def calculate_ambiguity_score(self, ambiguities: Dict) -> float:
        """
        Calculate overall ambiguity score
        """
        score = 0.0

        if ambiguities['low_confidence']:
            score += 0.3

        if ambiguities['vague_references']:
            score += 0.2 * len(ambiguities['vague_references'])

        if ambiguities['missing_information']:
            score += 0.3 * len(ambiguities['missing_information'])

        if ambiguities['multiple_interpretations']:
            score += 0.2 * len(ambiguities['multiple_interpretations'])

        return min(score, 1.0)
```

### 2. Clarification Strategies

Generating appropriate clarification requests:

```python
class ClarificationGenerator:
    def __init__(self):
        self.clarification_templates = {
            'navigation': [
                "Where exactly would you like me to go?",
                "Could you specify the destination more clearly?",
                "I need more specific directions to reach the destination."
            ],
            'manipulation': [
                "Which object would you like me to {action}?",
                "Could you point to the object you mean?",
                "I see multiple objects. Which one do you want me to {action}?"
            ],
            'location_ambiguous': [
                "Which {location_type} do you mean?",
                "Could you be more specific about the location?",
                "I know multiple locations like that. Can you clarify?"
            ],
            'vague_reference': [
                "Could you be more specific? What do you mean by '{vague_term}'?",
                "I need more details about '{vague_term}'.",
                "Can you clarify what you mean by '{vague_term}'?"
            ]
        }

    def generate_clarification_request(self, parsed_command: Dict, ambiguities: Dict) -> str:
        """
        Generate appropriate clarification request
        """
        intent = parsed_command.get('intent', 'unknown')

        # Check for vague references first
        if ambiguities['vague_references']:
            template = self.clarification_templates['vague_reference'][0]
            vague_term = ambiguities['vague_references'][0]
            return template.format(vague_term=vague_term)

        # Check for missing information
        if ambiguities['missing_information']:
            missing = ambiguities['missing_information'][0]

            if missing == 'destination' and intent == 'navigation':
                return self.clarification_templates['navigation'][0]
            elif missing == 'target_object' and intent == 'manipulation':
                action = self.get_action_from_intent(intent)
                template = self.clarification_templates['manipulation'][0]
                return template.format(action=action)

        # Default clarification
        return "I'm not sure I understood correctly. Could you please clarify your request?"

    def get_action_from_intent(self, intent: str) -> str:
        """
        Map intent to an appropriate action word
        """
        intent_actions = {
            'manipulation': 'manipulate',
            'navigation': 'navigate to',
            'social': 'interact with',
            'information': 'find'
        }
        return intent_actions.get(intent, 'perform action on')
```

## Integration with VLA Systems

### 1. Complete NLU Pipeline

```python
class RoboticNLU:
    def __init__(self):
        self.semantic_parser = SemanticParser()
        self.context_processor = ContextAwareNLU()
        self.multi_turn_processor = MultiTurnNLU()
        self.ambiguity_detector = AmbiguityDetector()
        self.clarification_generator = ClarificationGenerator()

        # For neural processing (optional)
        try:
            self.neural_parser = NeuralSemanticParser()
            self.use_neural = True
        except:
            self.use_neural = False

    def process_command(self, command: str, robot_context: Dict = None) -> Dict[str, Any]:
        """
        Complete NLU pipeline for robotic command processing
        """
        if robot_context is None:
            robot_context = {}

        # Step 1: Parse the command
        if self.use_neural:
            parsed_command = self.neural_parser.parse_command(command)
        else:
            parsed_command = self.semantic_parser.parse_command(command, robot_context)

        # Step 2: Process in multi-turn context
        contextual_command = self.multi_turn_processor.process_utterance(
            command, {**robot_context, **parsed_command}
        )

        # Step 3: Detect ambiguities
        ambiguities = self.ambiguity_detector.detect_ambiguity(contextual_command)

        # Step 4: Generate clarification if needed
        clarification = None
        if ambiguities['needs_clarification']:
            clarification = self.clarification_generator.generate_clarification_request(
                contextual_command, ambiguities
            )

        # Step 5: Prepare final output
        result = {
            'command': command,
            'parsed': contextual_command,
            'ambiguities': ambiguities,
            'clarification_needed': ambiguities['needs_clarification'],
            'clarification_request': clarification,
            'ready_for_execution': not ambiguities['needs_clarification'],
            'execution_plan': self.generate_execution_plan(contextual_command) if not ambiguities['needs_clarification'] else None
        }

        return result

    def generate_execution_plan(self, parsed_command: Dict) -> List[Dict[str, Any]]:
        """
        Generate high-level execution plan from parsed command
        """
        intent = parsed_command.get('intent')
        entities = parsed_command.get('entities', {})

        plan = []

        if intent == 'navigation':
            plan.append({
                'action': 'navigate',
                'target': entities.get('locations', ['unknown'])[0] if entities.get('locations') else 'unknown'
            })

        elif intent == 'manipulation':
            plan.append({
                'action': 'approach_object',
                'target': entities.get('objects', ['unknown'])[0] if entities.get('objects') else 'unknown'
            })
            plan.append({
                'action': 'grasp_object',
                'target': entities.get('objects', ['unknown'])[0] if entities.get('objects') else 'unknown'
            })

        elif intent == 'information':
            plan.append({
                'action': 'locate_object',
                'target': entities.get('objects', ['unknown'])[0] if entities.get('objects') else 'unknown'
            })

        return plan

# Example usage
def example_usage():
    nlu = RoboticNLU()

    # Example robot context
    robot_context = {
        'location': 'kitchen',
        'visible_objects': [
            {'name': 'red cup', 'type': 'cup', 'color': 'red', 'location': 'table'},
            {'name': 'blue bottle', 'type': 'bottle', 'color': 'blue', 'location': 'counter'}
        ],
        'capabilities': ['navigation', 'manipulation', 'speech']
    }

    # Test commands
    commands = [
        "Go to the living room",
        "Pick up the red cup",
        "It is dirty",  # Refers to last mentioned object
        "Take the bottle to the table"
    ]

    for cmd in commands:
        result = nlu.process_command(cmd, robot_context)
        print(f"Command: {cmd}")
        print(f"Intent: {result['parsed'].get('intent')}")
        print(f"Entities: {result['parsed'].get('entities')}")
        print(f"Clarification needed: {result['clarification_needed']}")
        if result['clarification_request']:
            print(f"Clarification: {result['clarification_request']}")
        print("---")

if __name__ == "__main__":
    example_usage()
```

## Performance Considerations

### 1. Efficiency Optimizations

```python
class EfficientNLU:
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 1000

    def process_command_cached(self, command: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process command with caching for efficiency
        """
        cache_key = self.create_cache_key(command, context)

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self.process_command(command, context)

        # Add to cache with size management
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = result

        return result

    def create_cache_key(self, command: str, context: Dict) -> str:
        """
        Create a cache key from command and simplified context
        """
        import hashlib
        context_key = str(sorted(context.items())) if context else ""
        full_key = f"{command}||{context_key}"
        return hashlib.md5(full_key.encode()).hexdigest()
```

## Acceptance Criteria Met

- [X] Command parsing techniques
- [X] Intent recognition examples
- [X] Error handling strategies