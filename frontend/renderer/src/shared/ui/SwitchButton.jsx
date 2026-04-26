export function SwitchButton({ checked, disabled, onClick }) {
  return (
    <button className={`switch-button ${checked ? "on" : ""}`} disabled={disabled} onClick={onClick}>
      <span />
    </button>
  );
}
