export function PipelinePanel({ active, activeTone = "good", children, className = "", rightMeta, title }) {
  return (
    <article className={`pipeline-column ${className}`}>
      <header className="column-header">
        <h2>{title}</h2>
        <div className="header-right">
          {rightMeta}
          <span className={`column-dot ${active ? activeTone : "idle"}`} />
        </div>
      </header>
      <div className="column-body">{children}</div>
    </article>
  );
}
