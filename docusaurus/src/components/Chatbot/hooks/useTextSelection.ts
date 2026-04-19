// Text Selection Hook - Detect and handle text selection on the page

import { useState, useEffect, useCallback } from 'react';
import { CHATBOT_CONFIG } from '../config';

interface UseTextSelectionReturn {
  selectedText: string | null;
  selectionPosition: { x: number; y: number } | null;
  clearSelection: () => void;
}

/**
 * Hook to detect text selection and trigger the "Ask AI" button
 * Uses native window.getSelection() API
 */
export function useTextSelection(): UseTextSelectionReturn {
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [selectionPosition, setSelectionPosition] = useState<{ x: number; y: number } | null>(null);

  const clearSelection = useCallback(() => {
    setSelectedText(null);
    setSelectionPosition(null);
    if (window.getSelection) {
      window.getSelection()?.removeAllRanges();
    }
  }, []);

  useEffect(() => {
    const handleSelectionChange = () => {
      const selection = window.getSelection();

      if (!selection || selection.isCollapsed) {
        setSelectedText(null);
        setSelectionPosition(null);
        return;
      }

      const text = selection.toString().trim();

      // Validate selection length
      if (!text || text.length === 0) {
        setSelectedText(null);
        setSelectionPosition(null);
        return;
      }

      if (text.length > CHATBOT_CONFIG.VALIDATION.MAX_SELECTED_TEXT_LENGTH) {
        // Truncate if too long
        setSelectedText(text.slice(0, CHATBOT_CONFIG.VALIDATION.MAX_SELECTED_TEXT_LENGTH));
      } else {
        setSelectedText(text);
      }

      // Get position for the floating button
      try {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        // Position above the selection, centered horizontally
        setSelectionPosition({
          x: rect.left + rect.width / 2,
          y: rect.top - 10, // 10px above
        });
      } catch {
        setSelectionPosition(null);
      }
    };

    // Use selectionchange event for modern browsers
    document.addEventListener('selectionchange', handleSelectionChange);

    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, []);

  // Expose a method to trigger the chat with selected text
  useEffect(() => {
    const handleTriggerChat = (event: CustomEvent<{ text: string }>) => {
      // Already handled by parent, but we can clear after
    };

    document.addEventListener('chatbot:trigger-select' as keyof WindowEventMap, handleTriggerChat as EventListener);

    return () => {
      document.removeEventListener('chatbot:trigger-select' as keyof WindowEventMap, handleTriggerChat as EventListener);
    };
  }, []);

  return {
    selectedText,
    selectionPosition,
    clearSelection,
  };
}

export default useTextSelection;