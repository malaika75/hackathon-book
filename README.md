# Physical AI & Humanoid Robotics Textbook

A comprehensive textbook covering the fundamental and advanced concepts of Physical AI and Humanoid Robotics, from ROS 2 fundamentals to Visual Language-Action (VLA) models.

## Overview

This textbook provides a structured learning path for students and practitioners interested in robotics, artificial intelligence, and human-robot interaction. It covers the complete spectrum of humanoid robotics, from basic communication frameworks to advanced AI-powered decision making.

## Modules

### Module 1: ROS 2 Fundamentals
Learn the Robot Operating System 2 (ROS 2), the foundational framework for robot software development.

**Chapters:**
- Introduction & Architecture
- CLI Tools, Packages & Client Libraries
- Data Types & Transformations
- Launch Files & Labs

**Topics Covered:**
- Nodes, topics, services, and actions
- Package and workspace management
- Python and C++ client libraries
- Coordinate transformations with tf2
- System orchestration

---

### Module 2: Robotics Simulation
Master robotic simulation environments for developing and testing robotic systems without physical hardware.

**Chapters:**
- Robotic Simulation Environments
- Robot Description & ROS Integration
- Physics, Sensors & Kinematics
- Labs - Robot Models & Sensors

**Topics Covered:**
- Gazebo and simulation platforms
- URDF and SDF robot descriptions
- Physics engine configuration
- Sensor simulation (lidar, camera, IMU)
- Forward and inverse kinematics

---

### Module 3: AI-Robot Brain
Explore the artificial intelligence aspects of robotics, including perception, navigation, and manipulation.

**Chapters:**
- Perception & SLAM
- Navigation, Path Planning & State Estimation
- Robot Manipulation & Labs

**Topics Covered:**
- Computer vision and perception systems
- Simultaneous Localization and Mapping (SLAM)
- ROS 2 Navigation Stack
- Path planning algorithms (A*, RRT)
- State estimation (Kalman filters, particle filters)
- Robot manipulation with MoveIt 2

---

### Module 4: VLA Models
Discover the cutting-edge intersection of large language models, computer vision, and robotic action.

**Chapters:**
- LLMs & Vision-Language Models
- VLA Integration & NLU
- Action Generation & Labs
- Case Studies & Ethics

**Topics Covered:**
- Transformer architecture and LLMs
- Vision-language models (VLMs)
- Natural language understanding for robotics
- Visual Language-Action integration
- Ethical considerations in AI robotics

## Prerequisites

- Basic programming knowledge (Python recommended)
- Understanding of linear algebra and calculus
- Familiarity with Linux command line

## Installation

### ROS 2 Setup
```bash
# Install ROS 2 Humble (recommended)
sudo apt update
sudo apt install ros-humble-desktop
```

### Python Dependencies
```bash
pip install numpy opencv-python cv-bridge
```

### Documentation Setup
```bash
# Install dependencies
npm install

# Start development server
npm run start
```

## Building the Book

```bash
# Build the documentation
npm run build

# Serve locally
npm run serve
```

## Lab Exercises

Each module includes hands-on lab exercises:

- **Lab 1**: ROS 2 Basic Communication
- **Lab 2**: ROS 2 Service & Action
- **Lab 3**: Robot Model in Simulation
- **Lab 4**: Sensor Integration & Data Acquisition
- **Lab 5**: SLAM and Navigation
- **Lab 6**: Object Detection & Grasping
- **Lab 7**: Natural Language Task Planning
- **Lab 8**: VLA for Environment Interaction

## Learning Outcomes

After completing this textbook, you will be able to:

1. Develop ROS 2 applications for robot communication
2. Create and simulate robot models in Gazebo
3. Implement perception and navigation systems
4. Apply AI techniques for robot control
5. Integrate natural language with robotic actions
6. Consider ethical implications in AI robotics

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This textbook is for educational purposes.

## RAG Backend API

This project includes a FastAPI backend with an intelligent RAG Agent for the Physical AI Textbook chatbot.

### Features

- **FastAPI Backend**: REST API server running on localhost:8000
- **RAG Agent**: Intelligent agent using OpenAI GPT-4o-mini and Qdrant vector database
- **Two Query Modes**:
  - `/chat` - Normal chat query
  - `/chat-with-context` - Contextual chat with selected text
- **Source Citations**: Answers include module, chapter, and section references

### Quick Start

```bash
# Navigate to backend
cd backend

python -m venv venv

venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Add your API keys to .env

# Start the server
uvicorn src.main:app --reload --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Normal chat query |
| `/chat-with-context` | POST | Contextual chat with selected text |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Normal chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ROS2?"}'

# Contextual chat
curl -X POST http://localhost:8000/chat-with-context \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain this", "selected_text": "ROS2 is the next generation of ROS"}'
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT model |
| `COHERE_API_KEY` | Cohere API key for embeddings |
| `QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Vector collection name (default: physical-ai-textbook) |

### Project Structure

```
backend/
├── src/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Settings loader
│   ├── api/
│   │   └── routes.py     # API endpoints
│   ├── agents/
│   │   └── rag_agent.py  # RAG agent logic
│   ├── services/
│   │   ├── embedding.py  # Cohere embedding client
│   │   └── retrieval.py  # Qdrant retrieval
│   └── models/
│       └── schemas.py    # Pydantic models
└── requirements.txt      # Python dependencies
```

### Ingestion

Ingest documents into the Qdrant vector database for the RAG chatbot.

```bash
# Navigate to backend
cd backend

# Activate virtual environment
venv\Scripts\activate

# Prepare documents
# Place .txt or .md files in ./data/docs/

# Run ingestion (API-limit-safe)
python scripts/ingest_qdrant.py

# Or use legacy script
python scripts/ingest_book.py
```

The ingestion script:
- Reads documents from `./data/docs/`
- Chunks text using LangChain (500 tokens, 50 overlap)
- Generates embeddings with Cohere embed-english-v3.0
- Uploads to Qdrant collection `physical-ai-textbook`
- Tracks progress to skip already-embedded chunks on re-run

### Environment Variables

| Variable | Description |
|----------|-------------|
| `COHERE_API_KEY` | Cohere API key for embeddings |
| `QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Collection name (default: physical-ai-textbook) |
| `DOCS_PATH` | Path to docs folder (default: ./data/docs) |

## RAG Chatbot Frontend

Integrate the RAG chatbot with the Docusaurus documentation site for interactive Q&A about the textbook.

### Features

- **Chat Interface**: ChatGPT-style conversational UI
- **Theme Support**: Light/dark mode that matches Docusaurus theme
- **Text Selection**: Select text on any page and click "Ask AI" to ask about it
- **Source Citations**: AI responses include relevant module/chapter references
- **Mobile Support**: Full-screen chat on mobile devices

### Quick Start

```bash
# Start Docusaurus
cd docusaurus
npm start
```

### Integration

Add the Chatbot to your pages using the wrapper:

```tsx
// In your page or layout
import Chatbot from '@site/src/components/Chatbot';

export default function MyLayout({ children }) {
  return (
    <>
      {children}
      <Chatbot />
    </>
  );
}
```

Or use the wrapper component:

```tsx
import ChatbotWrapper from '@site/src/components/Chatbot/wrappers/ChatbotWrapper';

export default function MyPage() {
  return (
    <ChatbotWrapper>
      <YourPageContent />
    </ChatbotWrapper>
  );
}
```

### Configuration

Edit `docusaurus/src/components/Chatbot/config.ts` to configure:

- `API_BASE_URL`: Backend URL (default: http://localhost:8000)
- `TIMEOUTS.REQUEST`: Request timeout (default: 10s)
- `DIMENSIONS.*`: UI dimensions

### Project Structure

```
docusaurus/src/components/Chatbot/
├── index.tsx              # Main component
├── ChatWindow.tsx        # Chat interface
├── FloatingButton.tsx   # Toggle button
├── MessageList.tsx      # Message display
├── styles.module.css     # Styling
├── types/index.ts       # TypeScript types
├── config.ts            # Configuration
├── services/
│   └── api.ts           # API service
├── hooks/
│   ├── useChat.ts      # Chat state
│   ├── useTheme.ts     # Theme detection
│   └── useTextSelection.ts  # Text selection
├── components/
│   └── SelectionButton.tsx   # "Ask AI" button
└── wrappers/
    └── ChatbotWrapper.tsx   # Page wrapper
```

### API Endpoints (from backend)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Normal chat query |
| `/chat-with-context` | POST | Chat with selected text |

## Acknowledgments

- ROS 2 Community
- Open Source Robotics Foundation
- NVIDIA Isaac Sim
- Gazebo Simulator Team

---

**Version:** 1.0
**Last Updated:** 2026
