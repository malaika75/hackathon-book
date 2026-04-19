---
id: 16
title: "Text Selection Chatbot Feature"
stage: "general"
date: "2026-04-19"
surface: "agent"
model: "oc/minimax-m2.5-free"
feature: "none"
branch: "003-rag-chatbot-frontend"
user: "unknown"
command: "/sp.phr"
labels: ["chatbot", "text-selection", "ui-feature", "rag"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "docusaurus/src/components/Chatbot/TextSelectionHandler.tsx"
  - "docusaurus/src/components/Chatbot/ChatWindow.tsx"
  - "docusaurus/src/components/Chatbot/wrappers/ChatbotWrapper.tsx"
tests: []
---

# Text Selection Chatbot Feature

## Prompt
--title "Implement text selection to chatbot feature" --stage general --feature none

## Response
I implemented a text selection feature in the chatbot frontend that works like ChatGPT:

1. Created `TextSelectionHandler.tsx` component that captures text selections globally on the page via mouseup events
2. Shows a floating "Ask AI" button above the selected text
3. Dispatches a custom event `chatbot:text-selected` with the selected text
4. Updated `ChatWindow.tsx` to capture selected text as context and pass it to the API via `sendChatWithContext`
5. Added the handler to `ChatbotWrapper.tsx`

The flow: User selects text on page → floating button appears → click sends text to chatbot as context.

## Evaluation
The feature is now complete and follows the ChatGPT pattern where selected text becomes the conversation context.

---

📝 PHR-016 recorded in history/prompts/general/

Stage: general
Feature: none
Files modified: 3
Tests involved: 0

Acceptance Criteria (PASS only if all true)
- [x] Full prompt preserved verbatim (no truncation)
- [x] Stage and routing determined correctly
- [x] Metadata fields populated; missing values noted explicitly