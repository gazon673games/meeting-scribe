import { AlertTriangle, X } from "lucide-react";

export function RuntimeNoticeTray({ notices, onDismiss }) {
  if (!notices?.length) {
    return null;
  }

  return (
    <div aria-live="polite" className="runtime-notice-tray" role="status">
      {notices.map((notice) => (
        <article key={notice.key} className={`runtime-notice ${notice.severity === "error" ? "error" : "warning"}`}>
          <AlertTriangle className="runtime-notice-icon" size={15} />
          <div className="runtime-notice-copy">
            <div className="runtime-notice-title">
              <strong>{notice.title}</strong>
              {notice.count > 1 ? <span>x{notice.count}</span> : null}
            </div>
            <p title={notice.message}>{notice.message}</p>
            {notice.detail ? <small title={notice.detail}>{notice.detail}</small> : null}
          </div>
          <button aria-label="Dismiss runtime notice" type="button" onClick={() => onDismiss(notice.key)}>
            <X size={14} />
          </button>
        </article>
      ))}
    </div>
  );
}
