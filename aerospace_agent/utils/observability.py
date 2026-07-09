"""结构化日志 + Metrics 观测性模块。

第一性原理：
  1. 日志是事件流——每条日志是一个结构化事件（JSON），含 timestamp/level/module/event/data
  2. Metrics 是可聚合的数值——counter（计数）、timer（耗时）、gauge（瞬时值）
  3. 零外部依赖：stdlib logging + json + time + threading
  4. 可通过 AEROSPACE_LOG_LEVEL 环境变量控制级别，AEROSPACE_LOG_JSON=1 输出 JSON
  5. Metrics 快照可随时导出，供 TUI/CLI/外部监控系统消费
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# 结构化日志
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """JSON 结构化日志格式器。"""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        # 附加结构化数据
        data = getattr(record, "data", None)
        if data and isinstance(data, dict):
            entry["data"] = data
        # 异常信息
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


class StructuredLogger:
    """结构化日志器——封装 logging.Logger，支持 event + data 模式。

    用法::

        log = get_logger("agent")
        log.event("tool_call", data={"tool": "propagate_orbit", "engine": "poliastro"})
        log.event("llm_response", data={"tokens": 350, "latency_ms": 1200})
    """

    def __init__(self, name: str, logger: logging.Logger):
        self._name = name
        self._logger = logger

    def event(self, event: str, level: str = "INFO",
              data: Optional[Dict[str, Any]] = None) -> None:
        """记录结构化事件。"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        extra = {"event": event, "data": data or {}}
        self._logger.log(log_level, event, extra=extra)

    def debug(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.event(event, level="DEBUG", data=data)

    def info(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.event(event, level="INFO", data=data)

    def warning(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.event(event, level="WARNING", data=data)

    def error(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.event(event, level="ERROR", data=data)

    def exception(self, event: str,
                  data: Optional[Dict[str, Any]] = None) -> None:
        """记录异常事件（含 traceback）。"""
        extra = {"event": event, "data": data or {}}
        self._logger.exception(event, extra=extra)


_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.Lock()
_logging_configured = False


def _configure_root_logging() -> None:
    """配置根日志处理器（幂等）。"""
    global _logging_configured
    if _logging_configured:
        return
    with _loggers_lock:
        if _logging_configured:
            return
        level_name = os.environ.get("AEROSPACE_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_name, logging.WARNING)
        use_json = os.environ.get("AEROSPACE_LOG_JSON", "0") == "1"

        handler = logging.StreamHandler(sys.stderr)
        if use_json:
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        handler.setLevel(level)

        root = logging.getLogger("aerospace_agent")
        root.setLevel(level)
        root.addHandler(handler)
        root.propagate = False
        _logging_configured = True


def get_logger(name: str) -> StructuredLogger:
    """获取结构化日志器。

    Args:
        name: 日志器名称（如 "agent", "loop", "adapter.poliastro"）
    Returns:
        StructuredLogger 实例
    """
    _configure_root_logging()
    full_name = f"aerospace_agent.{name}" if not name.startswith("aerospace_agent") else name
    with _loggers_lock:
        if full_name not in _loggers:
            _loggers[full_name] = StructuredLogger(
                full_name, logging.getLogger(full_name))
        return _loggers[full_name]


# ---------------------------------------------------------------------------
# Metrics 收集
# ---------------------------------------------------------------------------

class MetricsCollector:
    """线程安全的 Metrics 收集器——counter / timer / gauge。

    用法::

        metrics = get_metrics()
        metrics.inc("tool_calls", tags={"tool": "propagate_orbit"})
        with metrics.timer("llm_latency"):
            response = llm.chat(...)
        metrics.gauge("context_tokens", 3500)
        snapshot = metrics.snapshot()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"value": 0, "tags": {}})
        self._timers: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_s": 0.0, "min_s": float("inf"),
                     "max_s": 0.0, "tags": {}})
        self._gauges: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"value": 0.0, "tags": {}})

    def inc(self, name: str, value: int = 1,
            tags: Optional[Dict[str, str]] = None) -> None:
        """递增计数器。"""
        with self._lock:
            key = self._tag_key(name, tags)
            entry = self._counters[key]
            entry["value"] += value
            entry["name"] = name
            if tags:
                entry["tags"] = tags

    def gauge(self, name: str, value: float,
              tags: Optional[Dict[str, str]] = None) -> None:
        """设置仪表值（瞬时值）。"""
        with self._lock:
            key = self._tag_key(name, tags)
            entry = self._gauges[key]
            entry["value"] = value
            entry["name"] = name
            if tags:
                entry["tags"] = tags

    def timer(self, name: str,
              tags: Optional[Dict[str, str]] = None):
        """上下文管理器——计时代码块。

        用法::

            with metrics.timer("llm_latency"):
                response = llm.chat(...)
        """
        return _TimerContext(self, name, tags)

    def record_timing(self, name: str, duration_s: float,
                      tags: Optional[Dict[str, str]] = None) -> None:
        """直接记录耗时。"""
        with self._lock:
            key = self._tag_key(name, tags)
            entry = self._timers[key]
            entry["count"] += 1
            entry["total_s"] += duration_s
            entry["min_s"] = min(entry["min_s"], duration_s)
            entry["max_s"] = max(entry["max_s"], duration_s)
            entry["name"] = name
            if tags:
                entry["tags"] = tags

    def snapshot(self) -> Dict[str, Any]:
        """导出当前 metrics 快照。"""
        with self._lock:
            counters = []
            for key, entry in self._counters.items():
                counters.append({
                    "name": entry.get("name", key),
                    "value": entry["value"],
                    "tags": entry.get("tags", {}),
                })
            timers = []
            for key, entry in self._timers.items():
                count = entry["count"]
                total = entry["total_s"]
                timers.append({
                    "name": entry.get("name", key),
                    "count": count,
                    "total_s": round(total, 6),
                    "avg_s": round(total / count, 6) if count else 0,
                    "min_s": round(entry["min_s"], 6) if count else 0,
                    "max_s": round(entry["max_s"], 6) if count else 0,
                    "tags": entry.get("tags", {}),
                })
            gauges = []
            for key, entry in self._gauges.items():
                gauges.append({
                    "name": entry.get("name", key),
                    "value": entry["value"],
                    "tags": entry.get("tags", {}),
                })
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "counters": counters,
                "timers": timers,
                "gauges": gauges,
            }

    def reset(self) -> None:
        """重置所有 metrics。"""
        with self._lock:
            self._counters.clear()
            self._timers.clear()
            self._gauges.clear()

    @staticmethod
    def _tag_key(name: str, tags: Optional[Dict[str, str]]) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"


class _TimerContext:
    """计时上下文管理器。"""

    def __init__(self, collector: MetricsCollector, name: str,
                 tags: Optional[Dict[str, str]]):
        self._collector = collector
        self._name = name
        self._tags = tags
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start
        self._collector.record_timing(self._name, duration, self._tags)
        return False


_metrics: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """获取全局 MetricsCollector 单例。"""
    global _metrics
    if _metrics is not None:
        return _metrics
    with _metrics_lock:
        if _metrics is None:
            _metrics = MetricsCollector()
        return _metrics


__all__ = [
    "StructuredLogger",
    "MetricsCollector",
    "get_logger",
    "get_metrics",
]
