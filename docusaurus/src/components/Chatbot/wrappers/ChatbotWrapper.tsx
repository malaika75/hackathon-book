// Chatbot Wrapper for Docusaurus Layout
// This component wraps Docusaurus pages and includes the Chatbot

import React, { ReactNode } from 'react';
import Chatbot from '@site/src/components/Chatbot';
import { TextSelectionHandler } from '@site/src/components/Chatbot/TextSelectionHandler';

interface ChatbotWrapperProps {
  children: ReactNode;
}

/**
 * Wrapper component that adds the Chatbot to any Docusaurus page.
 * Use this in your layout or page components.
 *
 * Usage in docusaurus/src/pages/*.tsx:
 * ```tsx
 * import ChatbotWrapper from '@site/src/components/Chatbot/wrappers/ChatbotWrapper';
 *
 * export default function MyPage() {
 *   return (
 *     <ChatbotWrapper>
 *       <YourPageContent />
 *     </ChatbotWrapper>
 *   );
 * }
 * ```
 *
 * Or in your custom theme's Layout.tsx:
 * ```tsx
 * import ChatbotWrapper from '@site/src/components/Chatbot/wrappers/ChatbotWrapper';
 *
 * export default function Layout(props) {
 *   return (
 *     <ChatbotWrapper>
 *       <OriginalLayout {...props} />
 *     </ChatbotWrapper>
 *   );
 * }
 * ```
 */
export function ChatbotWrapper({ children }: ChatbotWrapperProps): React.JSX.Element {
  return (
    <>
      {children}
      <TextSelectionHandler />
      <Chatbot />
    </>
  );
}

export default ChatbotWrapper;