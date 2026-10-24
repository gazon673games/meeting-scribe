import { Send } from 'lucide-react';

export const AssistantPromptBox = ({
  disabled,
  submitDisabled,
  text,
  onSubmit,
  onTextChange,
}) => (
  <div className="assistant-input">
    <textarea
      disabled={disabled}
      placeholder="Ask the assistant anything..."
      value={text}
      onChange={(event) => onTextChange(event.target.value)}
    />
    <button disabled={submitDisabled || !text.trim()} onClick={onSubmit}>
      <Send size={15} />
      Send
    </button>
  </div>
);
