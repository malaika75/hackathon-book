# Research: RAG Chatbot Frontend Integration

**Feature**: 003-rag-chatbot-frontend
**Date**: 2026-04-17

## Research Questions

### 1. Docusaurus Custom Component Integration

**Question**: How to add custom React components to Docusaurus pages?

**Findings**:
- Docusaurus supports custom React components in `src/components/` directory
- Components can be imported in MDX files or used via `docusaurus.config.ts`
- Use `useColorMode` hook from `@docusaurus/theme-common` for theme detection
- Client-side components should use `client` directive in MDX or wrap with `BrowserOnly`

**Decision**: Use `src/components/Chatbot/` directory with React components, integrate via MDX or theme customizations.

### 2. Theme Detection (Light/Dark Mode)

**Question**: How to detect and respond to Docusaurus theme changes?

**Findings**:
- `@docusaurus/theme-common` provides `useColorMode` hook
- Hook returns `{ colorMode, setColorMode, toggleColorMode }`
- CSS custom properties: `--ifm-color-var-*` for theme-aware styling
- Theme changes trigger re-render when using the hook

**Decision**: Use `useColorMode` hook for reactive theme detection.

### 3. Text Selection Context Menu

**Question**: How to implement "Ask AI about this" on text selection?

**Findings**:
- Use `window.getSelection()` to detect selected text
- Add custom context menu via `document.addEventListener('contextmenu', ...)`
- Alternative: floating button near selection using `Selection.getRangeAt(0).getBoundingClientRect()`
- Chrome doesn't support custom context menu items, use floating button approach

**Decision**: Implement floating button that appears when text is selected.

### 4. API Integration with Backend

**Question**: How to integrate with RAG backend (002-rag-agent-api)?

**Findings**:
- Backend exposes `/chat` and `/chat-with-context` endpoints
- Use `fetch` or `axios` for HTTP requests
- Handle CORS - backend already configured for frontend access
- Response format includes `answer` and `sources` fields

**Decision**: Use `fetch` API for backend communication.

## Alternatives Considered

| Approach | Rationale |
|----------|------------|
| Custom Docusaurus plugin | Overkill for single component |
| iframe chat window | Loses theme integration |
| External chat service | Doesn't integrate with RAG backend |
| React component (chosen) | Best balance of integration and simplicity |

## Summary

All research questions resolved. Implementation approach:
1. React component in `docusaurus/src/components/Chatbot/`
2. `useColorMode` hook for theme detection
3. Floating button near text selection for contextual queries
4. Fetch API for backend communication