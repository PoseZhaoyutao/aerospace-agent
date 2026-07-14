import { Compass, Moon, Plus, Search, Settings, Sun } from "lucide-react";
import type { ThreadSummary } from "../types";

type Props = {
  threads: ThreadSummary[];
  selectedId: string | null;
  query: string;
  dark: boolean;
  onQuery: (value: string) => void;
  onNewChat: () => void;
  onSelect: (thread: ThreadSummary) => void;
  onTheme: () => void;
};

export function Sidebar({ threads, selectedId, query, dark, onQuery, onNewChat, onSelect, onTheme }: Props) {
  const filtered = threads.filter((thread) => thread.title.toLowerCase().includes(query.toLowerCase()));
  return (
    <aside className="sidebar" data-layout="sidebar">
      <div className="brand-row">
        <div className="brand-mark" aria-label="Aerospace Agent"><Compass size={20} strokeWidth={2.6} /></div>
        <span className="brand-name">Aerospace</span>
        <span className="brand-dot" />
      </div>
      <button className="new-chat" onClick={onNewChat}><Plus size={17} /> New chat</button>
      <label className="search-box">
        <Search size={16} />
        <input aria-label="Search sessions" value={query} onChange={(event) => onQuery(event.target.value)} placeholder="Search" />
      </label>
      <div className="sidebar-rule" />
      <div className="session-heading"><span>Sessions</span><span className="session-count">{threads.length}</span></div>
      <div className="session-list">
        {filtered.map((thread) => (
          <button key={thread.thread_id} className={`session-item ${selectedId === thread.thread_id ? "selected" : ""}`} onClick={() => onSelect(thread)}>
            <span className="session-icon" />
            <span>{thread.title}</span>
          </button>
        ))}
        {!filtered.length && <p className="no-sessions">No sessions yet.</p>}
      </div>
      <div className="sidebar-footer">
        <button className="footer-button" onClick={onTheme}><Settings size={16} /> Settings <span className="footer-theme">{dark ? <Moon size={14} /> : <Sun size={14} />}</span></button>
      </div>
    </aside>
  );
}
