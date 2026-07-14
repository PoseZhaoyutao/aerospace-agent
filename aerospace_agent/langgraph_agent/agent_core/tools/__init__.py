"""Self-contained Agent Core tool services and stable catalog."""

from .files import DEFAULT_IMPORTANT_PATHS, FileService
from .browser import (
    BrowserService,
    build_playwright_navigation_adapter,
    build_playwright_screenshot_adapter,
)
from .terminal import TerminalService
from .web import SearchProvider, WebService
from .system import (
    CORE_TOOL_NAMES,
    CoreToolCatalog,
    CoreToolServices,
    ToolCatalogEntry,
    build_core_tool_catalog,
)

__all__ = [
    "CORE_TOOL_NAMES",
    "CoreToolCatalog",
    "CoreToolServices",
    "DEFAULT_IMPORTANT_PATHS",
    "BrowserService",
    "SearchProvider",
    "WebService",
    "FileService",
    "TerminalService",
    "build_playwright_screenshot_adapter",
    "build_playwright_navigation_adapter",
    "ToolCatalogEntry",
    "build_core_tool_catalog",
]
