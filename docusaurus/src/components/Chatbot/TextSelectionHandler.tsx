// Text Selection Handler with native onclick

import React, { useState, useEffect, useRef } from 'react';

interface SelectionState {
  text: string;
  x: number;
  y: number;
}

export function TextSelectionHandler(): React.JSX.Element | null {
  const [selection, setSelection] = useState<SelectionState | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleMouseUp = () => {
      const sel = window.getSelection();
      const text = sel?.toString().trim() || '';

      if (!text || text.length < 2) {
        setSelection(null);
        return;
      }

      try {
        const range = sel!.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        setSelection({
          text: text,
          x: rect.left + rect.width / 2,
          y: rect.top - 10,
        });
      } catch {
        setSelection(null);
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      // Don't hide if clicking on our button
      if (buttonRef.current && buttonRef.current.contains(e.target as Node)) return;
      setSelection(null);
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mousedown', handleMouseDown);
    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  // Use effect to attach native onclick when button is rendered
  useEffect(() => {
    if (buttonRef.current && selection) {
      buttonRef.current.onclick = function() {
        console.log('[Handler] BUTTON CLICKED!', selection.text);
        (window as any).__SELECTED_TEXT = selection.text;
        const evt = new CustomEvent('text-selected-action', { detail: { text: selection.text } });
        document.dispatchEvent(evt);
        setSelection(null);
      };
    }
  }, [selection]);

  if (!selection) return null;

  return (
    <button
      ref={buttonRef}
      type="button"
      style={{
        position: 'fixed',
        left: selection.x,
        top: selection.y,
        transform: 'translate(-50%, -100%)',
        zIndex: 99999,
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '10px 20px',
        background: 'linear-gradient(135deg, #10a37f 0%, #0d8c6d 100%)',
        color: 'white',
        border: '2px solid rgba(255, 255, 255, 0.3)',
        borderRadius: '12px',
        fontSize: '14px',
        fontWeight: 600,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        cursor: 'pointer',
        boxShadow: '0 4px 20px rgba(16, 163, 127, 0.4), 0 2px 8px rgba(0,0,0,0.2)',
        transition: 'all 0.2s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translate(-50%, -100%) scale(1.05)';
        e.currentTarget.style.boxShadow = '0 6px 24px rgba(16, 163, 127, 0.5), 0 4px 12px rgba(0,0,0,0.3)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translate(-50%, -100%) scale(1)';
        e.currentTarget.style.boxShadow = '0 4px 20px rgba(16, 163, 127, 0.4), 0 2px 8px rgba(0,0,0,0.2)';
      }}
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
      Ask AI
    </button>
  );
}

export default TextSelectionHandler;