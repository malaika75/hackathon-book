// Chatbot Component - Main Entry Point
// Provides a ChatGPT-style conversational interface for the textbook

import React, { useState, useEffect } from 'react';
import { FloatingButton } from './FloatingButton';
import { ChatWindow } from './ChatWindow';
import { TextSelectionHandler } from './TextSelectionHandler';
import styles from './styles.module.css';

interface ChatbotProps {
  initialOpen?: boolean;
}

// Track if handlers are set up
let handlersReady = false;

function setupChatbotHandlers(setSelectedText: (text: string) => void, setIsOpen: (open: boolean) => void) {
  if (handlersReady) return;
  handlersReady = true;

  console.log('[Chatbot] Setting up handlers...');

  // Handle new custom event
  document.addEventListener('text-selected-action', ((event: CustomEvent) => {
    const text = event.detail?.text;
    if (text) {
      console.log('[Chatbot] Got text from event:', text);
      setSelectedText(text);
      setIsOpen(true);
    }
  }) as EventListener);

  // Poll for window property
  const interval = setInterval(() => {
    const text = (window as any).__SELECTED_TEXT;
    if (text) {
      console.log('[Chatbot] Got text from __SELECTED_TEXT:', text);
      (window as any).__SELECTED_TEXT = null;
      setSelectedText(text);
      setIsOpen(true);
    }
  }, 100);

  return () => clearInterval(interval);
}

/**
 * Main Chatbot component - renders floating button and chat window.
 * Works on all pages when loaded via Docusaurus clientModules.
 */
export function Chatbot({ initialOpen = false }: ChatbotProps): React.JSX.Element {
  const [isOpen, setIsOpen] = useState(initialOpen);
  const [selectedText, setSelectedText] = useState<string | undefined>(undefined);

  const handleToggle = () => setIsOpen(prev => !prev);
  const handleClose = () => setIsOpen(false);
  const handleSelectedTextConsumed = () => setSelectedText(undefined);

  useEffect(() => {
    const cleanup = setupChatbotHandlers(setSelectedText, setIsOpen);
    return cleanup;
  }, []);

  console.log('[Chatbot] Rendering. isOpen:', isOpen, 'selectedText:', selectedText?.substring(0, 30));

  return (
    <div className={styles.chatContainer}>
      <TextSelectionHandler />
      {isOpen && (
        <ChatWindow
          onClose={handleClose}
          selectedText={selectedText}
          onSelectedTextConsumed={handleSelectedTextConsumed}
        />
      )}
      <FloatingButton onClick={handleToggle} isOpen={isOpen} />
    </div>
  );
}

// Default export
export default Chatbot;