import React from "react";
import { ArrowDown, ArrowUp, Columns3, Eye, EyeOff, RotateCcw } from "lucide-react";

export function ColumnLayoutMenu({ layout }) {
  const [open, setOpen] = React.useState(false);
  const menuRef = React.useRef(null);

  React.useEffect(() => {
    if (!open) {
      return undefined;
    }

    const closeOnOutsidePointer = (event) => {
      if (!menuRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };

    window.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => window.removeEventListener("pointerdown", closeOnOutsidePointer);
  }, [open]);

  if (!layout) {
    return null;
  }

  const hidden = new Set(layout.hiddenIds);

  return (
    <div className="column-menu" ref={menuRef}>
      <button className="icon-button" title="Columns" type="button" onClick={() => setOpen((current) => !current)}>
        <Columns3 size={16} />
      </button>

      {open ? (
        <div className="column-menu-popover">
          {layout.columns.map((column, index) => {
            const isHidden = hidden.has(column.id);
            const visibleCount = layout.visibleIds.length;
            return (
              <div className={`column-menu-row ${isHidden ? "hidden" : ""}`} key={column.id}>
                <button
                  className="column-menu-icon"
                  disabled={!isHidden && visibleCount <= 1}
                  title={isHidden ? "Show" : "Hide"}
                  type="button"
                  onClick={() => (isHidden ? layout.showColumn(column.id) : layout.hideColumn(column.id))}
                >
                  {isHidden ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
                <span>{column.label}</span>
                <button
                  className="column-menu-icon"
                  disabled={index === 0}
                  title="Move up"
                  type="button"
                  onClick={() => layout.moveColumn(column.id, -1)}
                >
                  <ArrowUp size={14} />
                </button>
                <button
                  className="column-menu-icon"
                  disabled={index === layout.columns.length - 1}
                  title="Move down"
                  type="button"
                  onClick={() => layout.moveColumn(column.id, 1)}
                >
                  <ArrowDown size={14} />
                </button>
              </div>
            );
          })}

          <button className="column-menu-reset" type="button" onClick={layout.resetLayout}>
            <RotateCcw size={14} />
            <span>Reset</span>
          </button>
        </div>
      ) : null}
    </div>
  );
}
