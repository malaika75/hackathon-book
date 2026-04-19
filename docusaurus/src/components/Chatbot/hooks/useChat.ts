// Custom Hook for Chat State Management

import { useState, useCallback, useEffect } from 'react';
import { ChatMessage, SourceCitation } from '../types';
import { sendChatMessage, sendChatWithContext, ApiError } from '../services/api';
import { CHATBOT_CONFIG } from '../config';

interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (content: string, selectedText?: string) => Promise<void>;
  clearMessages: () => void;
  addUserMessage: (content: string) => void;
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function loadFromStorage(): ChatMessage[] {
  try {
    const stored = sessionStorage.getItem(CHATBOT_CONFIG.STORAGE_KEYS.MESSAGES);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function saveToStorage(messages: ChatMessage[]): void {
  try {
    sessionStorage.setItem(
      CHATBOT_CONFIG.STORAGE_KEYS.MESSAGES,
      JSON.stringify(messages)
    );
  } catch {
    // Ignore storage errors
  }
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadFromStorage());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Save to storage whenever messages change
  useEffect(() => {
    saveToStorage(messages);
  }, [messages]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    try {
      sessionStorage.removeItem(CHATBOT_CONFIG.STORAGE_KEYS.MESSAGES);
    } catch {
      // Ignore storage errors
    }
  }, []);

  const addUserMessage = useCallback((content: string) => {
    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
  }, []);

  const sendMessage = useCallback(async (content: string, selectedText?: string) => {
    // Add user message first
    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    setError(null);
    setIsLoading(true);

    try {
      let response;
      if (selectedText && selectedText.trim()) {
        response = await sendChatWithContext({
          question: content,
          selectedText: selectedText,
        });
      } else {
        response = await sendChatMessage({ question: content });
      }

      // Handle empty response
      const answerText = response?.answer || 'No response received';
      const sourcesList = response?.sources || [];

      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content: answerText,
        timestamp: new Date().toISOString(),
        sources: sourcesList,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      let errorMessage = 'An unexpected error occurred';
      console.error('Chat error:', err);

      if (err instanceof ApiError) {
        if (err.isNetworkError) {
          errorMessage = 'Unable to connect to the chat service. Please check your connection.';
        } else if (err.statusCode === 500) {
          errorMessage = 'Server error. Please try again later.';
        } else {
          errorMessage = err.message;
        }
      } else if (err instanceof TypeError) {
        // Network/CORS error
        errorMessage = 'Unable to connect to the chat service. Please check your connection.';
      }

      setError(errorMessage);

      // Add error message as assistant message
      const errorMsg: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content: errorMessage,
        timestamp: new Date().toISOString(),
        sources: [],
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    addUserMessage,
  };
}

export default useChat;