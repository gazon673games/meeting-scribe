import React from "react";

export function ModelMetadataGrid({ rows }) {
  return (
    <dl className="model-metadata-grid">
      {rows.map(([label, value]) => (
        <React.Fragment key={label}>
          <dt>{label}</dt>
          <dd title={String(value)}>{String(value)}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}
