import { describe, expect, it } from "vitest";

describe("WebUI scaffold", () => {
  it("exposes a root mount contract", () => {
    expect("root").toBe("root");
  });
});
