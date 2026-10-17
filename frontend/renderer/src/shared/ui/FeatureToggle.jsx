export function FeatureToggle({ checked, description, disabled, label, onClick }) {
  return (
    <button aria-pressed={Boolean(checked)} className={`feature-toggle ${checked ? "selected" : ""}`} disabled={disabled} title={description} onClick={onClick}>
      <span>{label}{description ? <small>{description}</small> : null}</span>
      <b />
    </button>
  );
}
