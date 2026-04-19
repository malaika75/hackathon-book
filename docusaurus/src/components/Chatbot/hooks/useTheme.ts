// Theme Detection Hook using Docusaurus useColorMode

import { useColorMode } from '@docusaurus/theme-common';
import { ChatTheme } from '../types';

interface UseThemeReturn {
  theme: ChatTheme;
  isDark: boolean;
}

/**
 * Hook to detect and respond to Docusaurus theme changes
 * Uses Docusaurus's built-in useColorMode hook
 */
export function useChatTheme(): UseThemeReturn {
  const { isDarkTheme } = useColorMode();

  return {
    theme: isDarkTheme ? 'dark' : 'light',
    isDark: isDarkTheme,
  };
}

export default useChatTheme;