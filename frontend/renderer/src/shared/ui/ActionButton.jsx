export const ActionButton = ({ disabled, icon, label, onClick }) => (
  <button
    type="button"
    aria-label={label}
    className="action-button"
    disabled={disabled}
    title={label}
    onClick={onClick}
  >
    {icon}
    <span>{label}</span>
  </button>
);
