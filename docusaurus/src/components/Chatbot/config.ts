// src/config.ts - SIMPLE & SAFE VERSION
export const CHATBOT_CONFIG = {
  // ✅ Local ke liye localhost, production ke liye HF URL
  // Abhi ke liye hardcode kar rahe hain — baad mein env variable se replace kar sakte hain
  API_BASE_URL: 'https://malaika909-rag-backend.hf.space',
  
  API_ENDPOINTS: {
    CHAT: '/chat',
    CHAT_WITH_CONTEXT: '/chat-with-context',
  },
  
  TIMEOUTS: {
    REQUEST: 60000,
    THEME_SWITCH: 500,
  },
  
  DIMENSIONS: {
    WINDOW_WIDTH: 400,
    WINDOW_HEIGHT: 500,
    BUTTON_SIZE: 56,
    MOBILE_BREAKPOINT: 768,
  },
  
  VALIDATION: {
    MAX_MESSAGE_LENGTH: 4000,
    MAX_SELECTED_TEXT_LENGTH: 2000,
  },
  
  STORAGE_KEYS: {
    MESSAGES: 'chatbot_messages',
    PREFERENCES: 'chatbot_preferences',
  },
} as const;

export default CHATBOT_CONFIG;