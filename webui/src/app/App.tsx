import { AlertTriangle, ChevronDown, Circle, Menu, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { createThread, loadHistory, loadThreads } from "../api";
import { Composer } from "../components/Composer";
import { EmptyState } from "../components/EmptyState";
import { MessageTimeline } from "../components/MessageTimeline";
import { Sidebar } from "../components/Sidebar";
import { runMessage } from "../ws";
import type { HistoryMessage, TerminalEvent, ThreadSummary } from "../types";

const SELECTED_KEY = "aerospace-agent.selected-thread";

export function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [selected, setSelected] = useState<ThreadSummary | null>(null);
  const [messages, setMessages] = useState<HistoryMessage[]>([]);
  const [terminal, setTerminal] = useState<TerminalEvent | null>(null);
  const [query, setQuery] = useState("");
  const [dark, setDark] = useState(() => localStorage.getItem("aerospace-agent.theme") === "dark");
  const [connection, setConnection] = useState<"connecting" | "connected" | "disconnected" | "error">("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("aerospace-agent.theme", dark ? "dark" : "light");
  }, [dark]);

  const selectThread = async (thread: ThreadSummary) => {
    setSelected(thread);
    localStorage.setItem(SELECTED_KEY, JSON.stringify({ project_id: thread.project_id, thread_id: thread.thread_id }));
    setTerminal(null);
    try {
      setMessages((await loadHistory(thread.thread_id)).messages);
      setError(null);
    } catch (reason) {
      setMessages([]);
      localStorage.removeItem(SELECTED_KEY);
      setError(reason instanceof Error ? reason.message : "Unable to load history");
    }
  };

  useEffect(() => {
    void loadThreads().then((loaded) => {
      setThreads(loaded);
      const stored = localStorage.getItem(SELECTED_KEY);
      const selectedId = stored ? (JSON.parse(stored) as { thread_id?: string }).thread_id : undefined;
      const restored = loaded.find((thread) => thread.thread_id === selectedId);
      if (restored) void selectThread(restored);
    }).catch((reason) => setError(reason instanceof Error ? reason.message : "Unable to connect"));
  }, []);

  const submit = async (message: string) => {
    setError(null);
    let thread = selected;
    if (!thread) {
      thread = await createThread();
      setThreads((current) => [...current, thread!]);
      setSelected(thread);
      localStorage.setItem(SELECTED_KEY, JSON.stringify({ project_id: thread.project_id, thread_id: thread.thread_id }));
    }
    setMessages((current) => [...current, { role: "user", content: message }]);
    setTerminal(null);
    setRunning(true);
    runMessage(thread.thread_id, message, {
      onConnection: setConnection,
      onError: setError,
      onTerminal: (event) => { setTerminal(event); setRunning(false); setConnection("disconnected"); },
    });
  };

  const hasMessages = messages.length > 0 || terminal !== null;
  const statusLabel = running ? "Running" : connection === "error" ? "Connection error" : "Ready";
  const filteredCount = useMemo(() => threads.filter((thread) => thread.title.toLowerCase().includes(query.toLowerCase())).length, [threads, query]);

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar threads={threads} selectedId={selected?.thread_id ?? null} query={query} dark={dark} onQuery={setQuery} onTheme={() => setDark((value) => !value)} onSelect={(thread) => void selectThread(thread)} onNewChat={() => { setSelected(null); setMessages([]); setTerminal(null); localStorage.removeItem(SELECTED_KEY); }} />
      <main className="main-panel">
        <header className="topbar"><button className="mobile-menu" onClick={() => setCollapsed((value) => !value)} aria-label="Toggle sidebar">{collapsed ? <Menu size={19} /> : <X size={19} />}</button><span className="topbar-title">{selected?.title ?? "New conversation"}</span><div className="topbar-status"><Circle size={8} fill="currentColor" /> {statusLabel}</div></header>
        {error && <div className="error-banner" role="alert"><AlertTriangle size={16} /> {error}</div>}
        {hasMessages ? <section className="conversation" data-layout="conversation"><MessageTimeline messages={messages} terminal={terminal} /><Composer disabled={running} onSubmit={submit} /></section> : <EmptyState disabled={running} onSubmit={submit} />}
        <footer className="status-strip" data-layout="runtime-status"><span><span className={`status-light ${running ? "busy" : ""}`} /> {statusLabel}</span><span>{filteredCount} session{filteredCount === 1 ? "" : "s"}</span><ChevronDown size={14} /></footer>
      </main>
    </div>
  );
}
