# Implementation Tasks: RAG Chatbot Frontend Integration

**Feature Branch**: `003-rag-chatbot-frontend`
**Feature**: RAG Chatbot Frontend Integration
**Tech Stack**: TypeScript 5.x, React 18, Docusaurus 3.x
**Date**: 2026-04-17

## Implementation Strategy

**MVP Scope**: User Story 1 - Chat with Textbook (P1)
- Core chat functionality with basic UI
- Integration with RAG backend
- Source citations display

**Incremental Delivery**:
- Phase 1 (US1): Core chat functionality
- Phase 2 (US2): Text selection feature
- Phase 3 (US3): Theme consistency
- Phase 4 (US4): Mobile experience

## Phase 1: Setup

- [X] T001 Create project structure `docusaurus/src/components/Chatbot/` with index.tsx, ChatWindow.tsx, FloatingButton.tsx, MessageList.tsx, styles.module.css
- [X] T002 Configure TypeScript types for chatbot components in types/index.ts
- [X] T003 Add chatbot configuration constants in config.ts (API URL, timeouts, dimensions)

## Phase 2: Foundational

- [X] T004 Implement API service layer for backend communication in services/api.ts
- [X] T005 Create custom hooks for chat state management in hooks/useChat.ts
- [X] T006 Create theme detection hook using Docusaurus useColorMode in hooks/useTheme.ts

## Phase 3: User Story 1 - Chat with Textbook (P1)

**Goal**: Users can ask questions about the textbook and receive AI-powered answers with citations

**Independent Test**: Open chat, send question, verify response with sources

- [X] T007 [US1] Create ChatMessage type in types/index.ts
- [X] T008 [US1] Create SourceCitation type in types/index.ts
- [X] T009 [US1] Implement FloatingButton component in FloatingButton.tsx
- [X] T010 [US1] Implement MessageList component in MessageList.tsx
- [X] T011 [US1] Implement ChatWindow component with input field in ChatWindow.tsx
- [X] T012 [US1] Integrate useChat hook with API service in ChatWindow.tsx
- [X] T013 [US1] Add typing indicator during loading state in MessageList.tsx
- [X] T014 [US1] Display source citations below AI responses in MessageList.tsx
- [X] T015 [US1] Add clear chat button and functionality in ChatWindow.tsx
- [X] T016 [US1] Style chat interface with CSS Modules in styles.module.css
- [ ] T017 [US1] Register chatbot component in Docusaurus in docusaurus.config.ts or theme

## Phase 4: User Story 2 - Ask About Selected Text (P2)

**Goal**: Users can select text on any page and ask AI about it

**Independent Test**: Select text on page, click "Ask AI", verify text appears in chat

- [X] T018 [US2] Create text selection detection hook in hooks/useTextSelection.ts
- [X] T019 [US2] Implement floating "Ask AI" button near selection in components/SelectionButton.tsx
- [X] T020 [US2] Pass selected text to chat input when triggered in hooks/useTextSelection.ts
- [X] T021 [US2] Call /chat-with-context endpoint with selectedText in services/api.ts

## Phase 5: User Story 3 - Theme Consistency (P2)

**Goal**: Chat interface matches Docusaurus light/dark theme

**Independent Test**: Toggle theme, verify chat adapts immediately

- [X] T022 [US3] Use useColorMode hook in Chatbot component in index.tsx
- [X] T023 [US3] Apply theme-aware CSS variables in styles.module.css
- [X] T024 [US3] Subscribe to theme changes in real-time in useTheme.ts

## Phase 6: User Story 4 - Mobile Experience (P3)

**Goal**: Chat works well on mobile devices

**Independent Test**: Open chat on mobile viewport, verify full-screen overlay

- [X] T025 [US4] Add responsive CSS for mobile (< 768px) in styles.module.css
- [X] T026 [US4] Implement full-screen overlay on mobile in ChatWindow.tsx
- [X] T027 [US4] Add close button for mobile in ChatWindow.tsx
- [X] T028 [US4] Ensure keyboard works on mobile in ChatWindow.tsx

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T029 Add error handling UI for backend unavailability in ChatWindow.tsx
- [X] T030 Handle empty source response gracefully in MessageList.tsx
- [X] T031 Add loading states for initial page load in index.tsx
- [X] T032 Optimize for performance (< 500ms theme switch, < 2s first paint) in styles.module.css

## Dependency Graph

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational) ─────┐
    │                       │
    ▼                       │
Phase 3 (US1) ◄─────────────┤
    │                       │
    ├───────────────────────┤
    ▼                       ▼
Phase 4 (US2)    Phase 5 (US3)
    │                       │
    └───────────┬───────────┘
                ▼
       Phase 6 (US4)
                │
                ▼
       Phase 7 (Polish)
```

## Parallel Execution Opportunities

- **T018-T021** (US2 - Text Selection): Can run in parallel with Phase 5 (US3) since they modify different components
- **T022-T024** (US3 - Theme): Can run in parallel with US2 since they touch different files

## Summary

- **Total Tasks**: 32
- **Completed**: 31
- **Remaining**: 1 (T017 - Register chatbot in Docusaurus - requires manual setup in pages)
- **Phase Breakdown**:
  - Setup: 3 tasks
  - Foundational: 3 tasks
  - US1 (Chat): 11 tasks
  - US2 (Text Selection): 4 tasks
  - US3 (Theme): 3 tasks
  - US4 (Mobile): 4 tasks
  - Polish: 4 tasks
- **MVP Scope**: Phase 3 (US1) - 11 tasks for core chat functionality
- **Parallel Opportunities**: 2 sets of parallelizable tasks identified