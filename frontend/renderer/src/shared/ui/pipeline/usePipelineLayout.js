import React from 'react';
import {
  readJsonStorage,
  removeStorage,
  writeStorage,
} from '../../lib/browserStorage';

const STORAGE_KEY = 'meeting-scribe.pipeline-layout.v1';

const normalizeOrder = (order, defaultOrder) => {
  const known = new Set(defaultOrder);
  const stored = Array.isArray(order)
    ? [...new Set(order.filter((id) => known.has(id)))]
    : [];
  return [...stored, ...defaultOrder.filter((id) => !stored.includes(id))];
};

const normalizeHidden = (hidden, order) => {
  const known = new Set(order);
  const next = Array.isArray(hidden)
    ? [...new Set(hidden.filter((id) => known.has(id)))]
    : [];
  return next.length >= order.length ? [] : next;
};

const readLayout = (storageKey) => {
  const parsed = readJsonStorage(storageKey, {});
  return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
    ? parsed
    : {};
};

export const usePipelineLayout = (columns, storageKey = STORAGE_KEY) => {
  const defaultOrder = React.useMemo(
    () => columns.map((column) => column.id),
    [columns],
  );
  const columnMap = React.useMemo(
    () => new Map(columns.map((column) => [column.id, column])),
    [columns],
  );
  const [layout, setLayout] = React.useState(() => readLayout(storageKey));
  const [resetRevision, setResetRevision] = React.useState(0);

  const order = normalizeOrder(layout.order, defaultOrder);
  const hiddenIds = normalizeHidden(layout.hidden, order);
  const hiddenSet = React.useMemo(() => new Set(hiddenIds), [hiddenIds]);
  const orderedColumns = order.map((id) => columnMap.get(id)).filter(Boolean);
  const visibleColumns = orderedColumns.filter(
    (column) => !hiddenSet.has(column.id),
  );
  const visibleIds = visibleColumns.map((column) => column.id);

  React.useEffect(() => {
    writeStorage(storageKey, JSON.stringify({ order, hidden: hiddenIds }));
  }, [hiddenIds, order, storageKey]);

  const moveColumn = React.useCallback(
    (id, direction) => {
      setLayout((current) => {
        const nextOrder = normalizeOrder(current.order, defaultOrder);
        const index = nextOrder.indexOf(id);
        const target = index + Number(direction || 0);
        if (index < 0 || target < 0 || target >= nextOrder.length) {
          return current;
        }
        const reordered = [...nextOrder];
        [reordered[index], reordered[target]] = [
          reordered[target],
          reordered[index],
        ];
        return { ...current, order: reordered };
      });
    },
    [defaultOrder],
  );

  const moveColumnTo = React.useCallback(
    (id, targetId, placement = 'before') => {
      setLayout((current) => {
        if (id === targetId) {
          return current;
        }

        const nextOrder = normalizeOrder(current.order, defaultOrder);
        const sourceIndex = nextOrder.indexOf(id);
        const targetIndex = nextOrder.indexOf(targetId);
        if (sourceIndex < 0 || targetIndex < 0) {
          return current;
        }

        const reordered = nextOrder.filter((columnId) => columnId !== id);
        const cleanTargetIndex = reordered.indexOf(targetId);
        const insertIndex = cleanTargetIndex + (placement === 'after' ? 1 : 0);
        reordered.splice(insertIndex, 0, id);
        return { ...current, order: reordered };
      });
    },
    [defaultOrder],
  );

  const hideColumn = React.useCallback(
    (id) => {
      setLayout((current) => {
        const nextOrder = normalizeOrder(current.order, defaultOrder);
        const nextHidden = new Set(normalizeHidden(current.hidden, nextOrder));
        if (nextOrder.length - nextHidden.size <= 1) {
          return current;
        }
        nextHidden.add(id);
        return { ...current, hidden: [...nextHidden] };
      });
    },
    [defaultOrder],
  );

  const showColumn = React.useCallback((id) => {
    setLayout((current) => {
      const nextHidden = new Set(current.hidden || []);
      nextHidden.delete(id);
      return { ...current, hidden: [...nextHidden] };
    });
  }, []);

  const resetLayout = React.useCallback(() => {
    removeStorage(storageKey);
    setLayout({ order: defaultOrder, hidden: [] });
    setResetRevision((current) => current + 1);
  }, [defaultOrder, storageKey]);

  return {
    columns: orderedColumns,
    hiddenIds,
    resetLayout,
    resetRevision,
    visibleColumns,
    visibleIds,
    hideColumn,
    moveColumn,
    moveColumnTo,
    showColumn,
  };
};
