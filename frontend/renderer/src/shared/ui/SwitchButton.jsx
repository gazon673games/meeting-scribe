export const SwitchButton = ({ checked, disabled, onClick, label }) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    aria-label={label}
    className={`switch-button ${checked ? 'on' : ''}`}
    disabled={disabled}
    onClick={onClick}
  >
    <span />
  </button>
);
