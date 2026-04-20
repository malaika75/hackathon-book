// Chatbot Configuration Constants

export const CHATBOT_CONFIG = {
  // API Configuration - Change this to your backend URL in production
  API_BASE_URL: 'http://localhost:8000',
  API_ENDPOINTS: {
    CHAT: '/chat',
    CHAT_WITH_CONTEXT: '/chat-with-context',
  },
  // Timeouts (in milliseconds)
  TIMEOUTS: {
    REQUEST: 60000, // 60 seconds - backend takes 15-30s for embedding + LLM
    THEME_SWITCH: 500, // 500ms per SC-005
  },
  // UI Dimensions
  DIMENSIONS: {
    WINDOW_WIDTH: 400,
    WINDOW_HEIGHT: 500,
    BUTTON_SIZE: 56,
    MOBILE_BREAKPOINT: 768,
  },
  // Validation
  VALIDATION: {
    MAX_MESSAGE_LENGTH: 4000,
    MAX_SELECTED_TEXT_LENGTH: 2000,
  },
  // Storage Keys
  STORAGE_KEYS: {
    MESSAGES: 'chatbot_messages',
    PREFERENCES: 'chatbot_preferences',
  },
} as const;

export default CHATBOT_CONFIG;