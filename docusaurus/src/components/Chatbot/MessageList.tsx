// Message List Component - Displays chat messages with sources

import React, { useRef, useEffect } from 'react';
import { ChatMessage, SourceCitation } from './types';
import styles from './styles.module.css';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
}

function SourceCitationComponent({ sources }: { sources: SourceCitation[] }) {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className={styles.sources}>
      <div className={styles.sourcesTitle}>Sources:</div>
      {sources.map((source, index) => (
        <div key={index} className={styles.sourceItem}>
          <span className={styles.sourceTitle}>{source.title}</span>
          {source.section && (
            <span className={styles.sourceSection}> - {source.section}</span>
          )}
          {source.relevanceScore && (
            <span className={styles.sourceScore}>
              ({Math.round(source.relevanceScore * 100)}% match)
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className={`${styles.message} ${styles.messageAssistant} ${styles.typingIndicator}`}>
      <div className={styles.typingDots}>
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  );
}

export function MessageList({ messages, isLoading }: MessageListProps): React.JSX.Element {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return (
      <div className={styles.emptyState}>
        <p>Chat with the textbook</p>
        <p className={styles.emptyHint}>Ask any question about the content</p>
      </div>
    );
  }

  return (
    <div className={styles.messageList}>
      {messages.map((message) => (
        <div
          key={message.id}
          className={`${styles.message} ${
            message.role === 'user' ? styles.messageUser : styles.messageAssistant
          }`}
        >
          <div className={styles.messageContent}>
            {message.content}
          </div>
          {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
            <SourceCitationComponent sources={message.sources} />
          )}
        </div>
      ))}
      {isLoading && <TypingIndicator />}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;