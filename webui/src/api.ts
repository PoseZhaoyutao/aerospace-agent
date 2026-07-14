import type { HistoryResponse, ThreadSummary } from "./types";

export async function loadThreads(): Promise<ThreadSummary[]> {
  const response = await fetch("/api/v1/threads");
  if (!response.ok) throw new Error(`Unable to load sessions (${response.status})`);
  return ((await response.json()) as { threads: ThreadSummary[] }).threads;
}

export async function createThread(title?: string): Promise<ThreadSummary> {
  const response = await fetch("/api/v1/threads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(title ? { title } : {}),
  });
  if (!response.ok) throw new Error(`Unable to create session (${response.status})`);
  return (await response.json()) as ThreadSummary;
}

export async function loadHistory(threadId: string): Promise<HistoryResponse> {
  const response = await fetch(`/api/v1/threads/${encodeURIComponent(threadId)}/history`);
  if (!response.ok) throw new Error(`Unable to load history (${response.status})`);
  return (await response.json()) as HistoryResponse;
}
