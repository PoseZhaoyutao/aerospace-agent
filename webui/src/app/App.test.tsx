import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

vi.mock("../api", () => ({
  loadThreads: vi.fn().mockResolvedValue([]),
  createThread: vi.fn().mockResolvedValue({ schema_version: "1.0.0", project_id: "p", thread_id: "t-1", title: "New chat" }),
  loadHistory: vi.fn().mockResolvedValue({ schema_version: "1.0.0", thread: {}, messages: [] }),
}));

describe("session shell", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => cleanup());

  it("shows the reference empty state and composer", async () => {
    render(<App />);
    expect(await screen.findByText("What should we")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Ask anything" })).toBeInTheDocument();
    expect(screen.queryByText("Apps")).not.toBeInTheDocument();
    expect(screen.queryByText("Skills")).not.toBeInTheDocument();
    expect(screen.queryByText("Automations")).not.toBeInTheDocument();
  });

  it("filters local sessions from the sidebar", async () => {
    const { loadThreads } = await import("../api");
    vi.mocked(loadThreads).mockResolvedValueOnce([
      { schema_version: "1.0.0", project_id: "p", thread_id: "a", title: "Orbital transfer" },
      { schema_version: "1.0.0", project_id: "p", thread_id: "b", title: "Thermal review" },
    ]);
    render(<App />);
    expect(await screen.findByText("Orbital transfer")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Search sessions" }), { target: { value: "thermal" } });
    expect(screen.queryByText("Orbital transfer")).not.toBeInTheDocument();
    expect(screen.getByText("Thermal review")).toBeInTheDocument();
  });
});
