export function ActionButton({ disabled, icon, label, onClick }) {
  return (
    <button aria-label={label} className="action-button" disabled={disabled} title={label} onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}
