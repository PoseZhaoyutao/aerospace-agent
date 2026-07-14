export type ThreadSummary = {
  schema_version: string;
  project_id: string;
  thread_id: string;
  title: string;
  created_at?: string | null;
  updated_at?: string | null;
  checkpoint_id?: string | null;
};

export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
  checkpoint_id?: string | null;
};

export type HistoryResponse = {
  schema_version: string;
  thread: ThreadSummary;
  messages: HistoryMessage[];
};

export type TerminalEvent = {
  type: "run.completed" | "run.interrupted" | "run.failed";
  status: "success" | "partial" | "interrupted" | "error" | "limit_reached" | "cycle_detected";
  reason_code?: "human_approval_required" | null;
  answer?: string;
  checkpoint_id?: string | null;
  citations?: Citation[];
};

export type Citation = {
  title?: string | null;
  excerpt?: string | null;
  score?: number | null;
  page_path?: string | null;
};
