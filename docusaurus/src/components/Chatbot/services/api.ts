// API Service Layer for RAG Backend Communication

import {
  ChatRequest,
  ChatResponse,
  ChatContextRequest,
} from '../types';
import { CHATBOT_CONFIG } from '../config';

class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public isNetworkError: boolean = false
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number = CHATBOT_CONFIG.TIMEOUTS.REQUEST
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timed out', undefined, true);
    }
    throw error;
  }
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const url = `${CHATBOT_CONFIG.API_BASE_URL}${CHATBOT_CONFIG.API_ENDPOINTS.CHAT}`;
  console.log('[Chatbot] Sending to:', url);

  try {
    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.detail || `HTTP error ${response.status}`,
        response.status
      );
    }

    return await response.json();
  } catch (error) {
    console.error('[Chatbot] API error:', error);
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof TypeError) {
      // Network error (CORS, DNS, etc.)
      console.error('[Chatbot] TypeError - likely network/CORS issue');
      throw new ApiError(
        'Unable to connect to the chat service. Please check your connection.',
        undefined,
        true
      );
    }
    throw new ApiError('An unexpected error occurred');
  }
}

export async function sendChatWithContext(
  request: ChatContextRequest
): Promise<ChatResponse> {
  const url = `${CHATBOT_CONFIG.API_BASE_URL}${CHATBOT_CONFIG.API_ENDPOINTS.CHAT_WITH_CONTEXT}`;

  try {
    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.detail || `HTTP error ${response.status}`,
        response.status
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof TypeError) {
      throw new ApiError(
        'Unable to connect to the chat service. Please check your connection.',
        undefined,
        true
      );
    }
    throw new ApiError('An unexpected error occurred');
  }
}

export { ApiError };
export default {
  sendChatMessage,
  sendChatWithContext,
  ApiError,
};