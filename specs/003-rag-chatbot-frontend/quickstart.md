# Quickstart: RAG Chatbot Frontend

**Feature**: 003-rag-chatbot-frontend

## Prerequisites

1. **Backend Running**: The RAG backend (002-rag-agent-api) must be running
   ```bash
   cd backend && uvicorn src.main:app --reload
   ```

2. **Docusaurus Running**: Start the Docusaurus development server
   ```bash
   cd docusaurus && npm start
   ```

## Setup

### 1. Create Chatbot Components

Create the following files in `docusaurus/src/components/Chatbot/`:

- `index.tsx` - Main chatbot component
- `ChatWindow.tsx` - Chat interface
- `FloatingButton.tsx` - Toggle button
- `MessageList.tsx` - Message display
- `styles.module.css` - Styling

### 2. Integrate into Docusaurus

Add to your page or theme:

```tsx
// In your page or custom component
import Chatbot from '@site/src/components/Chatbot';

export default function MyPage() {
  return (
    <div>
      <Chatbot />
    </div>
  );
}
```

### 3. Configure Backend URL

Set the backend API URL (default: `http://localhost:8000`):

```tsx
// In chatbot config
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';
```

## Usage

### Opening the Chat

Click the floating chat button (bottom-right corner) to open the chat window.

### Sending Messages

Type your question in the input field and press Enter or click Send.

### Asking About Selected Text

1. Select text on any textbook page
2. A floating "Ask AI" button appears near the selection
3. Click it to open chat with the selected text pre-filled

### Clearing Chat History

Click the "Clear" button in the chat header to reset the conversation.

## Testing

Run the Docusaurus development server:
```bash
cd docusaurus && npm start
```

Open http://localhost:3000 and test:
1. Click the chat button - window should open
2. Send a message - should receive a response with sources
3. Toggle dark/light mode - chat should adapt
4. Select text on a page - "Ask AI" button should appear

## Troubleshooting

- **CORS errors**: Ensure backend has CORS enabled for localhost:3000
- **No response**: Check backend is running on port 8000
- **Theme not updating**: Verify `useColorMode` hook is being used correctly