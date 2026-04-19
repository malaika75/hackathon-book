// Floating Button Component for Chat Toggle

import React from 'react';
import styles from './styles.module.css';

interface FloatingButtonProps {
  onClick: () => void;
  isOpen: boolean;
}

export function FloatingButton({ onClick, isOpen }: FloatingButtonProps): React.JSX.Element {
  return (
    <button
      className={`${styles.floatingButton} ${isOpen ? styles.floatingButtonOpen : ''}`}
      onClick={onClick}
      aria-label={isOpen ? 'Close chat' : 'Open chat'}
      title={isOpen ? 'Close chat' : 'Chat with the textbook'}
    >
      {isOpen ? (
        <svg
          width="24"
          height="24"
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
      ) : (
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      )}
    </button>
  );
}

export default FloatingButton;