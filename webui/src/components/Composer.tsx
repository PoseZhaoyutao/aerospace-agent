import { ArrowUp, Paperclip } from "lucide-react";
import { FormEvent, useState } from "react";

type Props = { disabled?: boolean; onSubmit: (message: string) => void };

export function Composer({ disabled, onSubmit }: Props) {
  const [value, setValue] = useState("");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const message = value.trim();
    if (!message || disabled) return;
    onSubmit(message);
    setValue("");
  };
  return (
    <form className="composer" data-layout="composer" onSubmit={submit}>
      <textarea aria-label="Ask anything" value={value} onChange={(event) => setValue(event.target.value)} placeholder="Ask anything..." rows={2} disabled={disabled} />
      <div className="composer-toolbar">
        <button type="button" className="icon-button" aria-label="Attachments unavailable" disabled><Paperclip size={17} /></button>
        <span className="model-chip">Qwen local</span>
        <span className="toolbar-spacer" />
        <span className="project-label">Local project</span>
        <button type="submit" className="send-button" aria-label="Send" disabled={disabled || !value.trim()}><ArrowUp size={18} /></button>
      </div>
    </form>
  );
}
