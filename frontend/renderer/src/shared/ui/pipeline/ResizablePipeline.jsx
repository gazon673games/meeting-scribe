import React from "react";
import { EyeOff } from "lucide-react";

const STORAGE_KEY = "meeting-scribe.pipeline-widths.v1";
const HANDLE_WIDTH = 7;

export function ResizablePipeline({ columns, onHideColumn, onReorderColumn, resetSignal = 0, storageKey = STORAGE_KEY }) {
  const containerRef = React.useRef(null);
  const dragRef = React.useRef(null);
  const hasCustomWidthsRef = React.useRef(hasStoredWidths(storageKey));
  const resetSignalRef = React.useRef(resetSignal);
  const columnsRef = React.useRef(columns);
  const draggedColumnRef = React.useRef("");
  const [dropTarget, setDropTarget] = React.useState(null);
  const [widths, setWidths] = React.useState(() => (hasCustomWidthsRef.current ? readStoredWidths(storageKey) : {}));
  const columnSignature = columns.map((column) => column.id).join("|");

  React.useEffect(() => {
    columnsRef.current = columns;
  }, [columnSignature]);

  React.useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    const syncWidths = () => {
      setWidths((current) => materializeWidths(columnsRef.current, hasCustomWidthsRef.current ? current : {}, container.clientWidth));
    };

    syncWidths();
    const observer = new ResizeObserver(syncWidths);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    setWidths((current) => materializeWidths(columns, hasCustomWidthsRef.current ? current : {}, container.clientWidth));
  }, [columnSignature]);

  React.useEffect(() => {
    if (resetSignalRef.current === resetSignal) {
      return;
    }
    resetSignalRef.current = resetSignal;
    hasCustomWidthsRef.current = false;
    window.localStorage?.removeItem(storageKey);
    const container = containerRef.current;
    setWidths(materializeWidths(columns, {}, container?.clientWidth || 0));
  }, [columnSignature, resetSignal, storageKey]);

  React.useEffect(() => {
    if (hasCustomWidthsRef.current && Object.keys(widths).length) {
      window.localStorage?.setItem(storageKey, JSON.stringify(widths));
    }
  }, [storageKey, widths]);

  React.useEffect(() => {
    const handlePointerMove = (event) => {
      const drag = dragRef.current;
      if (!drag) {
        return;
      }
      resizePair(drag.index, event.clientX - drag.startX, drag.startWidths);
    };

    const handlePointerUp = () => {
      dragRef.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
      handlePointerUp();
    };
  }, []);

  const resizePair = React.useCallback((index, delta, baseWidths = widths) => {
    const left = columnsRef.current[index];
    const right = columnsRef.current[index + 1];
    if (!left || !right) {
      return;
    }

    hasCustomWidthsRef.current = true;
    setWidths((current) => {
      const source = baseWidths || current;
      const leftWidth = safeWidth(source[left.id], left);
      const rightWidth = safeWidth(source[right.id], right);
      const pairWidth = leftWidth + rightWidth;
      const minLeft = Number(left.minWidth || 160);
      const minRight = Number(right.minWidth || 160);
      const maxLeft = Number(left.maxWidth || Number.POSITIVE_INFINITY);
      const maxRight = Number(right.maxWidth || Number.POSITIVE_INFINITY);
      const lower = Math.max(minLeft, pairWidth - maxRight);
      const upper = Math.min(maxLeft, pairWidth - minRight);
      const nextLeft = clamp(leftWidth + delta, lower, upper);

      return {
        ...current,
        [left.id]: Math.round(nextLeft),
        [right.id]: Math.round(pairWidth - nextLeft)
      };
    });
  }, [widths]);

  const gridTemplateColumns = columns
    .flatMap((column, index) => {
      const value = `${Math.round(safeWidth(widths[column.id], column))}px`;
      return index < columns.length - 1 ? [value, `${HANDLE_WIDTH}px`] : [value];
    })
    .join(" ");

  return (
    <section className="pipeline resizable-pipeline" ref={containerRef} style={{ gridTemplateColumns }}>
      {columns.map((column, index) => (
        <React.Fragment key={column.id}>
          {renderColumnElement(column, {
            headerProps: makeHeaderProps({
              column,
              dropTarget,
              draggedColumnId: draggedColumnRef.current,
              onReorderColumn,
              setDropTarget,
              draggedColumnRef
            }),
            layoutControls: makeLayoutControls({ column, columns, onHideColumn })
          })}
          {index < columns.length - 1 ? (
            <button
              aria-label={`Resize ${column.label} and ${columns[index + 1].label}`}
              className="column-resizer"
              title="Resize columns"
              type="button"
              onKeyDown={(event) => {
                if (event.key === "ArrowLeft") {
                  event.preventDefault();
                  resizePair(index, -16);
                }
                if (event.key === "ArrowRight") {
                  event.preventDefault();
                  resizePair(index, 16);
                }
              }}
              onPointerDown={(event) => {
                event.preventDefault();
                dragRef.current = {
                  index,
                  startX: event.clientX,
                  startWidths: widths
                };
                document.body.style.cursor = "col-resize";
                document.body.style.userSelect = "none";
              }}
            />
          ) : null}
        </React.Fragment>
      ))}
    </section>
  );
}

function renderColumnElement(column, injectedProps) {
  if (!React.isValidElement(column.element)) {
    return column.element;
  }
  return React.cloneElement(column.element, injectedProps);
}

function makeHeaderProps({ column, draggedColumnId, draggedColumnRef, dropTarget, onReorderColumn, setDropTarget }) {
  if (!onReorderColumn) {
    return null;
  }

  const isDropTarget = dropTarget?.id === column.id && dropTarget.sourceId !== column.id;
  return {
    className: [
      "column-header-draggable",
      draggedColumnId === column.id ? "dragging" : "",
      isDropTarget ? `drop-${dropTarget.placement}` : ""
    ]
      .filter(Boolean)
      .join(" "),
    draggable: true,
    onDragStart: (event) => {
      draggedColumnRef.current = column.id;
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", column.id);
    },
    onDragOver: (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      const placement = dropPlacementFromEvent(event);
      setDropTarget({ id: column.id, placement, sourceId: draggedColumnRef.current });
    },
    onDrop: (event) => {
      event.preventDefault();
      const sourceId = event.dataTransfer.getData("text/plain") || draggedColumnRef.current;
      const placement = dropPlacementFromEvent(event);
      if (sourceId && sourceId !== column.id) {
        onReorderColumn(sourceId, column.id, placement);
      }
      draggedColumnRef.current = "";
      setDropTarget(null);
    },
    onDragEnd: () => {
      draggedColumnRef.current = "";
      setDropTarget(null);
    }
  };
}

function makeLayoutControls({ column, columns, onHideColumn }) {
  if (!onHideColumn) {
    return null;
  }

  return (
    <div className="column-tools">
      <button
        aria-label={`Hide ${column.label}`}
        className="column-tool"
        disabled={!onHideColumn || columns.length <= 1}
        title="Hide"
        type="button"
        draggable={false}
        onClick={() => onHideColumn?.(column.id)}
        onDragStart={(event) => event.preventDefault()}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <EyeOff size={14} />
      </button>
    </div>
  );
}

function dropPlacementFromEvent(event) {
  const rect = event.currentTarget.getBoundingClientRect();
  return event.clientX > rect.left + rect.width / 2 ? "after" : "before";
}

function materializeWidths(columns, current, containerWidth) {
  const next = Object.fromEntries(columns.map((column) => [column.id, safeWidth(current[column.id], column)]));
  const available = Number(containerWidth || 0) - Math.max(0, columns.length - 1) * HANDLE_WIDTH;
  if (available <= 0) {
    return next;
  }

  const total = sumWidths(columns, next);
  const flexColumn = columns.find((column) => column.flex) || columns[Math.max(0, columns.length - 2)];

  if (available > total && flexColumn) {
    next[flexColumn.id] = Math.round(next[flexColumn.id] + available - total);
    return next;
  }

  if (available < total) {
    return fitWidthsToAvailable(columns, next, available);
  }

  return next;
}

function safeWidth(value, column) {
  const parsed = Number(value);
  const fallback = Number(column.defaultWidth || column.minWidth || 240);
  return Math.max(Number(column.minWidth || 160), Number.isFinite(parsed) && parsed > 0 ? parsed : fallback);
}

function sumWidths(columns, widths) {
  return columns.reduce((total, column) => total + safeWidth(widths[column.id], column), 0);
}

function fitWidthsToAvailable(columns, widths, available) {
  const minWidths = Object.fromEntries(columns.map((column) => [column.id, minWidth(column)]));
  const minTotal = columns.reduce((total, column) => total + minWidths[column.id], 0);
  if (minTotal >= available) {
    return minWidths;
  }

  const total = sumWidths(columns, widths);
  const shrinkNeeded = total - available;
  const shrinkable = total - minTotal;
  if (shrinkNeeded <= 0 || shrinkable <= 0) {
    return widths;
  }

  const ratio = Math.min(1, shrinkNeeded / shrinkable);
  const fitted = Object.fromEntries(
    columns.map((column) => {
      const width = safeWidth(widths[column.id], column);
      const min = minWidths[column.id];
      return [column.id, width - (width - min) * ratio];
    })
  );

  return balanceRoundedWidths(columns, fitted, available);
}

function balanceRoundedWidths(columns, widths, available) {
  const rounded = Object.fromEntries(columns.map((column) => [column.id, Math.round(widths[column.id])]));
  let diff = Math.round(available - columns.reduce((total, column) => total + rounded[column.id], 0));
  if (!diff) {
    return rounded;
  }

  const flexColumn = columns.find((column) => column.flex) || columns[columns.length - 1];
  const ordered = [flexColumn, ...columns.filter((column) => column.id !== flexColumn?.id)].filter(Boolean);
  for (const column of ordered) {
    if (!diff) {
      break;
    }
    if (diff > 0) {
      rounded[column.id] += diff;
      diff = 0;
      break;
    }

    const removable = Math.min(Math.abs(diff), Math.max(0, rounded[column.id] - minWidth(column)));
    rounded[column.id] -= removable;
    diff += removable;
  }
  return rounded;
}

function minWidth(column) {
  return Math.max(120, Number(column.minWidth || 160));
}

function clamp(value, min, max) {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function readStoredWidths(storageKey) {
  try {
    const raw = window.localStorage?.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function hasStoredWidths(storageKey) {
  try {
    return Boolean(window.localStorage?.getItem(storageKey));
  } catch {
    return false;
  }
}
