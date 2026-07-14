import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MessageTimeline } from "./MessageTimeline";

describe("message timeline", () => {
  it("renders bounded citation details with a terminal answer", () => {
    render(<MessageTimeline messages={[]} terminal={{
      type: "run.completed",
      status: "success",
      answer: "Verified answer",
      citations: [{ title: "Orbit notes", excerpt: "Evidence excerpt", score: 0.91, page_path: "knowledge/orbit.md" }],
    }} />);
    expect(screen.getByText("Verified answer")).toBeInTheDocument();
    expect(screen.getByText("Orbit notes")).toBeInTheDocument();
    expect(screen.getByText("Evidence excerpt")).toBeInTheDocument();
    expect(screen.getByText(/Evidence score 0\.91/)).toBeInTheDocument();
  });
});
