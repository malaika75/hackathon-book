// Chatbot Component - Main Entry Point
// Provides a ChatGPT-style conversational interface for the textbook

import React, { useState, useEffect } from 'react';
import { FloatingButton } from './FloatingButton';
import { ChatWindow } from './ChatWindow';
import styles from './styles.module.css';

interface ChatbotProps {
  initialOpen?: boolean;
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
  const handleTextSelected = (text: string) => {
    setSelectedText(text);
    setIsOpen(true);
  };
  const handleSelectedTextConsumed = () => setSelectedText(undefined);

  useEffect(() => {
    const handleTextSelect = (event: CustomEvent<{ text: string }>) => {
      if (event.detail?.text) handleTextSelected(event.detail.text);
    };
    document.addEventListener('chatbot:text-selected' as keyof WindowEventMap, handleTextSelect as EventListener);
    return () => {
      document.removeEventListener('chatbot:text-selected' as keyof WindowEventMap, handleTextSelect as EventListener);
    };
  }, []);

  return (
    <div className={styles.chatContainer}>
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