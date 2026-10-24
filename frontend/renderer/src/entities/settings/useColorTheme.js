import React from 'react';

import { normalizeTheme } from './model';
import { readStorage, writeStorage } from '../../shared/lib/browserStorage';

const THEME_STORAGE_KEY = 'meeting-scribe.color-theme.v1';

const readTheme = () => normalizeTheme(readStorage(THEME_STORAGE_KEY));

export const useColorTheme = ({ theme, persist = false } = {}) => {
  const [cachedTheme, setCachedTheme] = React.useState(readTheme);
  const colorTheme = theme === undefined ? cachedTheme : normalizeTheme(theme);

  React.useEffect(() => {
    if (!persist) return;
    writeStorage(THEME_STORAGE_KEY, colorTheme);
  }, [colorTheme, persist]);

  React.useEffect(() => {
    if (theme !== undefined) return undefined;
    const syncTheme = (event) => {
      if (event.key === THEME_STORAGE_KEY || event.key === null) {
        setCachedTheme(readTheme());
      }
    };
    window.addEventListener('storage', syncTheme);
    return () => window.removeEventListener('storage', syncTheme);
  }, [theme]);

  return colorTheme;
};
