import { classNames } from '../../shared/lib/classNames';
import React from 'react';
import { Activity, ChevronDown, ChevronRight, RefreshCw } from 'lucide-react';

import { meetingScribeClient } from '../../shared/api/meetingScribeClient';
import { displayDeviceLabel } from './deviceLabels';

const ProcessGroup = ({ group, selectedId, onSelect }) => {
  const [open, setOpen] = React.useState(true);
  const Icon = open ? ChevronDown : ChevronRight;

  return (
    <div className="process-group">
      <button
        className="process-group-head"
        type="button"
        onClick={() => setOpen((value) => !value)}
      >
        <Icon size={13} />
        <span>{displayDeviceLabel(group.label)}</span>
        <b>{group.sessions.length}</b>
      </button>
      {open ? (
        <div className="process-group-children">
          {group.sessions.map((session) => (
            <button
              className={classNames(
                'process-row',
                selectedId === session.id && 'selected',
              )}
              key={session.id}
              type="button"
              onClick={() => onSelect(session)}
            >
              <Activity size={13} className="process-row-icon" />
              <span className="process-row-label">{session.label}</span>
              {session.streams > 1 ? (
                <span className="process-row-badge">{session.streams}</span>
              ) : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
};

export const normalizeProcessGroups = (result) => {
  const groups =
    Array.isArray(result?.groups) && result.groups.length
      ? result.groups
      : [
          {
            id: 'active-apps',
            label: 'Active applications',
            sessions: result?.sessions || [],
          },
        ];

  return groups
    .map((group, index) => ({
      id: String(group.id || `group-${index}`),
      label: String(group.label || `Output ${index + 1}`),
      sessions: (group.sessions || [])
        .filter((session) => session?.id)
        .map((session) => ({
          ...session,
          groupLabel: session.groupLabel || group.label,
          fullLabel:
            session.fullLabel ||
            `${group.label || 'Application'} / ${session.label || session.pid}`,
        })),
    }))
    .filter((group) => group.sessions.length);
};

const ProcessCatalog = ({
  activeLoading,
  activeCatalog,
  groups,
  selectedId,
  onSelect,
}) => {
  if (activeLoading && activeCatalog === null) {
    return <div className="source-dropdown-empty">Loading...</div>;
  }
  if (groups.length === 0) {
    return (
      <div className="source-dropdown-empty">
        No applications with active audio found.
      </div>
    );
  }
  return (
    <div className="process-tree-list compact">
      {groups.map((group) => (
        <ProcessGroup
          group={group}
          key={group.id}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
};

export const SystemAudioPicker = ({
  catalog: externalCatalog,
  devices = [],
  disabled,
  error: externalError = '',
  loading: externalLoading = false,
  selected,
  selectedId,
  onRefresh,
  onSelect,
}) => {
  const rootRef = React.useRef(null);
  const buttonRef = React.useRef(null);
  const [open, setOpen] = React.useState(false);
  const [catalog, setCatalog] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');
  const [panelStyle, setPanelStyle] = React.useState(null);

  const activeCatalog = externalCatalog || catalog;
  const activeLoading = Boolean(externalLoading || loading);
  const activeError = externalError || error;
  const groups = React.useMemo(
    () => normalizeProcessGroups(activeCatalog),
    [activeCatalog],
  );

  const load = React.useCallback(
    (options = {}) => {
      if (onRefresh) {
        onRefresh(options);
        return;
      }
      setLoading(true);
      setError('');
      meetingScribeClient
        .request('list_process_sessions')
        .then((result) => setCatalog(result || { groups: [], sessions: [] }))
        .catch((requestError) =>
          setError(String(requestError?.message || requestError)),
        )
        .finally(() => setLoading(false));
    },
    [onRefresh],
  );

  const updatePanelPosition = React.useCallback(() => {
    const rect = buttonRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    const gap = 5;
    const margin = 12;
    const viewportWidth = window.innerWidth || 0;
    const viewportHeight = window.innerHeight || 0;
    const availableWidth = Math.max(160, viewportWidth - margin * 2);
    const width = Math.min(Math.max(rect.width, 320), availableWidth);
    const left = Math.max(
      margin,
      Math.min(rect.left, viewportWidth - width - margin),
    );
    const maxHeight = Math.max(
      180,
      viewportHeight - rect.bottom - margin - gap,
    );
    setPanelStyle({
      left: `${Math.round(left)}px`,
      maxHeight: `${Math.round(maxHeight)}px`,
      top: `${Math.round(rect.bottom + gap)}px`,
      width: `${Math.round(width)}px`,
    });
  }, []);

  React.useEffect(() => {
    if (open && activeCatalog === null && !activeLoading && !activeError) {
      load();
    }
  }, [activeCatalog, activeError, activeLoading, load, open]);

  React.useEffect(() => {
    if (!open) {
      return undefined;
    }
    const handlePointerDown = (event) => {
      if (!rootRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        setOpen(false);
      }
    };
    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  React.useLayoutEffect(() => {
    if (!open) {
      return undefined;
    }
    updatePanelPosition();
    window.addEventListener('resize', updatePanelPosition);
    window.addEventListener('scroll', updatePanelPosition, true);
    return () => {
      window.removeEventListener('resize', updatePanelPosition);
      window.removeEventListener('scroll', updatePanelPosition, true);
    };
  }, [open, updatePanelPosition]);

  const choose = async (item) => {
    if (!item || disabled) {
      return;
    }
    setOpen(false);
    try {
      await onSelect(item);
    } catch (selectionError) {
      setError(String(selectionError?.message || selectionError));
      console.warn('[audio inputs] source selection failed', selectionError);
    }
  };

  const selectedLabel = selected
    ? displayDeviceLabel(selected.fullLabel || selected.label)
    : 'No devices found';

  return (
    <div className="source-dropdown" ref={rootRef}>
      <button
        ref={buttonRef}
        className="source-dropdown-button"
        disabled={disabled}
        title={selectedLabel}
        type="button"
        onClick={() => setOpen((value) => !value)}
      >
        <span>{selectedLabel}</span>
        <ChevronDown size={14} />
      </button>

      {open ? (
        <div className="source-dropdown-panel" style={panelStyle || undefined}>
          <section className="source-dropdown-section">
            <div className="source-dropdown-title">System Outputs</div>
            {devices.length ? (
              devices.map((device) => (
                <button
                  className={classNames(
                    'source-option',
                    selectedId === device.id && 'selected',
                  )}
                  key={device.id}
                  type="button"
                  onClick={() => choose(device)}
                >
                  <span>{displayDeviceLabel(device.label)}</span>
                </button>
              ))
            ) : (
              <div className="source-dropdown-empty">
                No system outputs found.
              </div>
            )}
          </section>

          <section className="source-dropdown-section">
            <div className="process-tree-toolbar">
              <span className="source-dropdown-title">Applications</span>
              <button
                className="icon-button"
                disabled={activeLoading}
                title="Refresh app audio sources"
                type="button"
                onClick={() => load({ force: true })}
              >
                <RefreshCw size={12} className={activeLoading ? 'spin' : ''} />
              </button>
            </div>

            {activeError ? (
              <div className="process-tree-error">{activeError}</div>
            ) : null}

            <ProcessCatalog
              activeLoading={activeLoading}
              activeCatalog={activeCatalog}
              groups={groups}
              selectedId={selectedId}
              onSelect={choose}
            />
          </section>
        </div>
      ) : null}
    </div>
  );
};
