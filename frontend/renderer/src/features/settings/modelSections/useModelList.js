import React from "react";

export function joinDir(base, sub) {
  const left = String(base || "").trim();
  const right = String(sub || "").trim();
  if (!left && !right) return "";
  if (!left) return right;
  if (!right) return left;
  return `${left}/${right}`;
}

export function formatError(error) {
  return String(error?.message || error);
}

export function useModelList({ open, loadFn, refreshToken = 0 }) {
  const [models, setModels] = React.useState(null);
  const [error, setError] = React.useState("");

  const load = React.useCallback(() => {
    loadFn()
      .then((result) => setModels(result?.models || []))
      .catch((requestError) => setError(formatError(requestError)));
  }, [loadFn]);

  React.useEffect(() => {
    if (!open) return;
    load();
  }, [open, load]);

  React.useEffect(() => {
    if (!open || refreshToken <= 0) return;
    load();
  }, [open, refreshToken, load]);

  React.useEffect(() => {
    if (!models?.some((item) => item.downloading)) return undefined;
    const handle = window.setInterval(load, 1000);
    return () => window.clearInterval(handle);
  }, [models, load]);

  return { models, error, setError, load };
}

export async function runWithPending(setPending, key, action) {
  setPending((current) => new Set([...current, key]));
  try {
    await action();
  } finally {
    setPending((current) => {
      const next = new Set(current);
      next.delete(key);
      return next;
    });
  }
}
