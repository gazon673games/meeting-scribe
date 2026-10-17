import React from "react";

import { normalizeTheme } from "./model";

const THEME_STORAGE_KEY = "meeting-scribe.color-theme.v1";

function readTheme() {
  try {
    return normalizeTheme(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return "dark";
  }
}

export function useColorTheme({ theme, persist = false } = {}) {
  const [cachedTheme, setCachedTheme] = React.useState(readTheme);
  const colorTheme = theme === undefined ? cachedTheme : normalizeTheme(theme);

  React.useEffect(() => {
    if (!persist) return;
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, colorTheme);
    } catch {
      // Theme selection still works when local storage is unavailable.
    }
  }, [colorTheme, persist]);

  React.useEffect(() => {
    if (theme !== undefined) return;
    const syncTheme = (event) => {
      if (event.key === THEME_STORAGE_KEY || event.key === null) setCachedTheme(readTheme());
    };
    window.addEventListener("storage", syncTheme);
    return () => window.removeEventListener("storage", syncTheme);
  }, [theme]);

  return colorTheme;
}
