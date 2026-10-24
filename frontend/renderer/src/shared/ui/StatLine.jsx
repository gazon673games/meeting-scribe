export const StatLine = ({ accent = '', label, value }) => (
  <div className={`stat-line ${accent}`}>
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
);
