export function StatLine({ accent = "", label, value }) {
  return (
    <div className={`stat-line ${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
