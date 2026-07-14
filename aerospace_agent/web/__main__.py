"""Launch the local WebUI without coupling it to the task CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the Aerospace Agent WebUI")
    parser.add_argument("--workspace", default=".", help="workspace root")
    parser.add_argument("--host", default=None, help="bind host; defaults to WebUI settings")
    parser.add_argument("--port", type=int, default=None, help="bind port; defaults to WebUI settings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
    from aerospace_agent.langgraph_agent.config import load_settings
    from aerospace_agent.web.app import create_app
    from aerospace_agent.web.manager import AgentRuntimeManager

    workspace = Path(args.workspace).resolve()
    settings = load_settings(workspace=workspace)
    webui = settings.webui
    host = args.host or webui.host
    port = args.port or webui.port
    if webui.allow_lan or host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("WEBUI_LAN_NOT_ALLOWED: authenticated LAN access is not implemented")
    if not webui.enabled:
        raise SystemExit("WEBUI_DISABLED: WebUI is disabled")

    manager = AgentRuntimeManager(
        agent_factory=lambda: LangGraphAerospaceAgent(settings=settings),
        project_id=workspace.name,
        workspace_id=str(workspace),
    )
    app = create_app(
        manager=manager,
        static_dir=Path(__file__).with_name("static"),
        webui_enabled=webui.enabled,
        host=host,
        allow_lan=webui.allow_lan,
    )
    import uvicorn

    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
