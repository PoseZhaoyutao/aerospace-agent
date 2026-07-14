import type { TerminalEvent } from "./types";

type RunCallbacks = {
  onConnection?: (state: "connecting" | "connected" | "disconnected" | "error") => void;
  onTerminal: (event: TerminalEvent) => void;
  onError: (message: string) => void;
};

export function runMessage(
  threadId: string,
  message: string,
  callbacks: RunCallbacks,
): () => void {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/api/v1/ws`);
  const requestId = crypto.randomUUID();
  callbacks.onConnection?.("connecting");
  socket.onopen = () => {
    callbacks.onConnection?.("connected");
    socket.send(JSON.stringify({
      schema_version: "1.0.0",
      type: "run.start",
      request_id: requestId,
      thread_id: threadId,
      message,
    }));
  };
  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data) as Partial<TerminalEvent> & { type?: string };
    if (payload.type === "run.completed" || payload.type === "run.interrupted" || payload.type === "run.failed") {
      callbacks.onTerminal(payload as TerminalEvent);
      socket.close();
    }
  };
  socket.onerror = () => {
    callbacks.onConnection?.("error");
    callbacks.onError("The WebSocket connection failed");
  };
  socket.onclose = () => callbacks.onConnection?.("disconnected");
  return () => socket.close();
}
