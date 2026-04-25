export function ActionButton({ disabled, icon, label, onClick }) {
  return (
    <button className="action-button" disabled={disabled} onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}
