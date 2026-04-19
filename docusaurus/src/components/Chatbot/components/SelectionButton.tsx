// Selection Button Component - "Ask AI" button that appears near selected text

import React, { useState, useEffect, useRef } from 'react';
import styles from '../styles.module.css';

interface SelectionButtonProps {
  text: string;
  position: { x: number; y: number };
  onAskAI: (text: string) => void;
  onClose: () => void;
}

export function SelectionButton({
  text,
  position,
  onAskAI,
  onClose,
}: SelectionButtonProps): React.JSX.Element {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [style, setStyle] = useState<React.CSSProperties>({});

  // Calculate position (keep within viewport)
  useEffect(() => {
    const button = buttonRef.current;
    if (!button) return;

    const rect = button.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    let x = position.x - rect.width / 2;
    let y = position.y - rect.height - 10;

    // Keep within viewport horizontally
    if (x < 10) x = 10;
    if (x + rect.width > viewportWidth - 10) {
      x = viewportWidth - rect.width - 10;
    }

    // Keep within viewport vertically
    if (y < 10) y = position.y + 20; // Show below if no room above

    setStyle({
      position: 'fixed',
      left: x,
      top: y,
    });
  }, [position]);

  const handleClick = () => {
    onAskAI(text);
    onClose();
  };

  // Close on click outside or escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (buttonRef.current && !buttonRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    // Small delay to avoid triggering on the click that opened it
    setTimeout(() => {
      document.addEventListener('click', handleClickOutside);
    }, 100);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('click', handleClickOutside);
    };
  }, [onClose]);

  return (
    <button
      ref={buttonRef}
      className={styles.selectionButton}
      style={style}
      onClick={handleClick}
    >
      Ask AI
    </button>
  );
}

export default SelectionButton;