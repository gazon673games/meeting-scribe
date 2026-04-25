export function FeatureToggle({ checked, disabled, label, onClick }) {
  return (
    <button className={`feature-toggle ${checked ? "selected" : ""}`} disabled={disabled} onClick={onClick}>
      <span>{label}</span>
      <b />
    </button>
  );
}
