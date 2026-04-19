// Chatbot Component Types
// TypeScript type definitions for RAG Chatbot Frontend

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: SourceCitation[];
}

export interface SourceCitation {
  documentId: string;
  title: string;
  section: string;
  relevanceScore: number;
  url?: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceCitation[];
}

export interface ChatRequest {
  question: string;
}

export interface ChatContextRequest extends ChatRequest {
  selectedText: string;
}

export interface ChatSession {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  isOpen: boolean;
}

export interface UserPreferences {
  lastBackendUrl: string;
  theme: 'light' | 'dark' | 'system';
}

export type ChatTheme = 'light' | 'dark';