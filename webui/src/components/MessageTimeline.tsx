import type { HistoryMessage, TerminalEvent } from "../types";

type Props = { messages: HistoryMessage[]; terminal?: TerminalEvent | null };

export function MessageTimeline({ messages, terminal }: Props) {
  return (
    <div className="message-timeline" data-layout="message-timeline">
      {messages.map((message, index) => (
        <article className={`message ${message.role}`} key={`${message.checkpoint_id ?? "message"}-${index}`}>
          <div className="message-role">{message.role === "user" ? "YOU" : "AGENT"}</div>
          <div className="message-body">{message.content}</div>
        </article>
      ))}
      {terminal && <article className={`message assistant ${terminal.status}`}>
        <div className="message-role">AGENT / {terminal.status.replace("_", " ").toUpperCase()}</div>
        <div className="message-body">{terminal.answer || "The run ended without an answer."}</div>
        {terminal.reason_code && <div className="reason-badge">Human approval required in the existing Agent Core flow</div>}
      </article>}
    </div>
  );
}
