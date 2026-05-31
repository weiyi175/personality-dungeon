import json
import time
from pathlib import Path
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _default_log_path():
    name = f"metrics_{time.strftime('%Y-%m-%d')}.jsonl"
    return LOG_DIR / name


def log_metric(event_type: str, **kwargs) -> Path:
    """Append a single metric event to today's JSONL file.

    The event will always contain a millisecond `timestamp` and `event_type`.
    Returns the path written to.
    """
    event = {"timestamp": int(time.time() * 1000), "event_type": event_type}
    event.update(kwargs)
    path = _default_log_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def persist_metrics_batch(events, path: Path | None = None):
    """Write a batch of metric dictionaries to the log file.

    Each item in `events` should be JSON-serializable.
    """
    if path is None:
        path = _default_log_path()
    with path.open("a", encoding="utf-8") as f:
        for ev in events:
            if "timestamp" not in ev:
                ev["timestamp"] = int(time.time() * 1000)
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return path


@contextmanager
def timing_context(event_type: str, **meta):
    """Context manager to measure elapsed time and emit a metric on exit.

    Usage:
      with timing_context("rl_init", session_id=sid):
          do_work()
    """
    start = time.time()
    try:
        yield
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        meta.update({"latency_ms": duration_ms, "status": 500, "error": str(e)})
        log_metric(event_type, **meta)
        raise
    else:
        duration_ms = int((time.time() - start) * 1000)
        meta.update({"latency_ms": duration_ms, "status": meta.get("status", 200)})
        log_metric(event_type, **meta)


__all__ = ["log_metric", "persist_metrics_batch", "timing_context"]
