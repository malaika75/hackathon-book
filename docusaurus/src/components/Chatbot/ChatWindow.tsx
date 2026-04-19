// Chat Window Component - Main chat interface

import React, { useState, useRef, useEffect } from 'react';
import { MessageList } from './MessageList';
import { useChat } from './hooks/useChat';
import styles from './styles.module.css';

interface ChatWindowProps {
  onClose: () => void;
  selectedText?: string;
  onSelectedTextConsumed?: () => void;
}

export function ChatWindow({
  onClose,
  selectedText,
  onSelectedTextConsumed,
}: ChatWindowProps): React.JSX.Element {
  const { messages, isLoading, error, sendMessage, clearMessages } = useChat();
  const [inputValue, setInputValue] = useState('');
  const [contextText, setContextText] = useState<string | undefined>(undefined);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Pre-fill with selected text if available
  useEffect(() => {
    if (selectedText && selectedText.trim()) {
      setInputValue(selectedText);
      setContextText(selectedText); // Store as context for API call
      onSelectedTextConsumed?.();
    }
  }, [selectedText, onSelectedTextConsumed]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();

    const content = inputValue.trim();
    if (!content || isLoading) return;

    const context = contextText; // Capture context before clearing
    setInputValue('');
    setContextText(undefined); // Clear context after capturing
    await sendMessage(content, context);

    // Focus back on input after sending
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleClear = () => {
    clearMessages();
    inputRef.current?.focus();
  };

  return (
    <div className={styles.chatWindow}>
      <div className={styles.chatHeader}>
        <h3 className={styles.chatTitle}>Chat with Textbook</h3>
        <div className={styles.headerActions}>
          <button
            className={styles.clearButton}
            onClick={handleClear}
            title="Clear chat"
            disabled={messages.length === 0}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
            Clear
          </button>
          <button className={styles.closeButton} onClick={onClose} title="Close">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      </div>

      <MessageList messages={messages} isLoading={isLoading} />

      <form className={styles.inputArea} onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          className={styles.inputField}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          disabled={isLoading}
          rows={1}
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={!inputValue.trim() || isLoading}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>

      {error && <div className={styles.errorBanner}>{error}</div>}
    </div>
  );
}

export default ChatWindow;