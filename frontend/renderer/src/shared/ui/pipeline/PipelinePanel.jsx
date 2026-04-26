export function PipelinePanel({ active, activeTone = "good", children, className = "", headerControls, headerProps, rightMeta, showIndicator = false, title }) {
  const hasHeaderMeta = Boolean(headerControls || rightMeta || showIndicator);
  const { className: headerClassName = "", ...headerRestProps } = headerProps || {};

  return (
    <article className={`pipeline-column ${className}`}>
      <header {...headerRestProps} className={["column-header", headerClassName].filter(Boolean).join(" ")}>
        <h2>{title}</h2>
        {hasHeaderMeta ? (
          <div className="header-right">
            {headerControls}
            {rightMeta}
            {showIndicator ? <span className={`column-dot ${active ? activeTone : "idle"}`} /> : null}
          </div>
        ) : null}
      </header>
      <div className="column-body">{children}</div>
    </article>
  );
}
