import React from 'react';

export const joinDir = (base, sub) => {
  const left = String(base || '').trim();
  const right = String(sub || '').trim();
  if (!left && !right) return '';
  if (!left) return right;
  if (!right) return left;
  return `${left}/${right}`;
};

export const formatError = (error) => String(error?.message || error);

export const useModelList = ({ open, loadFn, refreshToken = 0 }) => {
  const [models, setModels] = React.useState(null);
  const [error, setError] = React.useState('');
  const scope = React.useRef({ active: false, loadFn: null, request: null });

  const load = React.useCallback(() => {
    const owner = scope.current;
    if (!owner.active || owner.loadFn !== loadFn) return Promise.resolve();
    if (owner.request) return owner.request.promise;
    const request = { promise: null };
    owner.request = request;
    setError('');
    request.promise = Promise.resolve()
      .then(loadFn)
      .then((result) => {
        if (!owner.active || owner.request !== request) return;
        setModels(Array.isArray(result?.models) ? result.models : []);
      })
      .catch((requestError) => {
        if (owner.active && owner.request === request) {
          setError(formatError(requestError));
        }
      })
      .finally(() => {
        if (owner.request === request) owner.request = null;
      });
    return request.promise;
  }, [loadFn]);

  React.useEffect(() => {
    const changed = scope.current.loadFn !== loadFn;
    const owner = { active: open, loadFn, request: null };
    scope.current = owner;
    if (open) {
      if (changed) setModels(null);
      load();
    }
    return () => {
      owner.active = false;
      owner.request = null;
    };
  }, [open, refreshToken, load, loadFn]);

  React.useEffect(() => {
    if (!open || !models?.some((item) => item.downloading)) return undefined;
    const handle = window.setInterval(load, 1000);
    return () => window.clearInterval(handle);
  }, [models, load, open]);

  return { models, error, setError, load };
};

export const runWithPending = async (setPending, key, action) => {
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
};
